
from __future__ import division
from data import load_dataset
import time
import argparse
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch
import itertools

# import multiprocessing
# cpus = multiprocessing.cpu_count()
# import ray

from support_func import  loss_permutation,\
                          loss_top_1_in_lat_top_k, normalize_numpy,\
                          get_nearestneighbors, sanitize, forward_pass, Normalize,\
                          get_nearestneighbors_partly, write_ivecs, save_transformed_data


def repeat(l, r):
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))


def pairwise_NNs_inner(x):
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I


def triplet_optimize(xt, xv, gt_nn, xq, gt, net, args, lambda_uniform, kpos, rank_negative, val_k, margin):
    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    N = gt_nn.shape[0]

    xt_var = torch.from_numpy(xt).to(args.device)

    # prepare optimizer

    if args.optim == "sgd":
        optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    elif args.optim == "adam":
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
    pdist = nn.PairwiseDistance(2)

    all_logs = []
    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]
        if args.optim == "sgd":
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        t0 = time.time()

        # Sample positives for triplet
        rank_pos = np.random.choice(kpos, size=N)
        positive_idx = gt_nn[np.arange(N), rank_pos]

        # Sample negatives for triplet
        net.eval()
        print("  Forward pass")
        xl_net = forward_pass(net, xt, 1024)
        print("  Distances")
        I = get_nearestneighbors(xl_net, xl_net, rank_negative, args.device, needs_exact=False)
        negative_idx = I[:, -1]

        # training pass
        print("  Train")
        net.train()
        # net_for_query.train()
        avg_triplet, avg_uniform, avg_loss = 0, 0, 0
        offending = idx_batch = 0

        # process dataset in a random order
        perm = np.random.permutation(N)

        t1 = time.time()

        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            n = i1 - i0

            data_idx = perm[i0:i1]

            # anchor, positives, negatives
            ins = xt_var[data_idx]
            pos = xt_var[positive_idx[data_idx]]
            neg = xt_var[negative_idx[data_idx]]

            # do the forward pass (+ record gradients)
            ins, pos, neg = net(ins), net(pos), net(neg)

            # triplet loss
            per_point_loss = pdist(ins, pos) - pdist(ins, neg) + margin
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()

            # entropy loss
            I = pairwise_NNs_inner(ins.data)
            distances = pdist(ins, ins[I])
            loss_uniform = - torch.log(distances).mean()

            # combined loss
            loss = loss_triplet + lambda_uniform * loss_uniform

            # collect some stats
            avg_triplet += loss_triplet.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_batch += 1

        avg_triplet /= idx_batch
        avg_uniform /= idx_batch
        avg_loss /= idx_batch

        logs = {
            'epoch': epoch,
            'loss_triplet': avg_triplet,
            'loss_uniform': avg_uniform,
            'loss': avg_loss,
            'offending': offending,
            'lr': args.lr
        }
        all_logs.append(logs)

        t2 = time.time()
        # maybe perform a validation run

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            yt = forward_pass(net, xt, 1024)
            yv = forward_pass(net, xv, 1024)
            logs['perm'] = loss_permutation(xt, yt, args, k=val_k, size=10**4)

            logs['train_top1_k'] = loss_top_1_in_lat_top_k(xt, xt, yt, yt, args, 2, val_k, size=10**5, name="TRAIN")
            logs['valid_top1_k'] = loss_top_1_in_lat_top_k(xv, xt, yv, yt, args, 1, val_k, size=10**5, name="VALID")

            yq = forward_pass(net, xq, 1024)
            logs['query_top1_k'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, val_k, size=10**4, name="QUERY_tr")
            logs['query_top1_2k'] = loss_top_1_in_lat_top_k(xq, xt, yq, yt, args, 1, 2*val_k, size=10**4, name="QUERY_tr")

            net.train()

        t3 = time.time()

        # synthetic logging
        print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' lr = %f'
               ' loss = %g = %g + lam * %g, offending %d' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            args.lr,
            avg_loss, avg_triplet, avg_uniform, offending
        ))

        logs['times'] = (t1 - t0, t2 - t1, t3 - t2)

    return all_logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ray.init(num_cpus=cpus)

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('dataset options')
    aa("--database", default="deep1b") # can be "bigann", "deep1b" or "*.fvecs"
    aa("--size_base", type=int, default=int(1e6),
       help="size of evaluation dataset")
    aa("--num_learn", type=int, default=int(5e5),
       help="nb of learning vectors")

    group = parser.add_argument_group('Model hyperparameters')
    aa("--dint", type=int, default=1024,
       help="size of hidden states")
    aa("--dout", type=int, default=16,
       help="output dimension")
    aa("--lambda_uniform", type=float, default=0.05,
       help="weight of the uniformity loss")

    group = parser.add_argument_group('Training hyperparameters')
    aa("--batch_size", type=int, default=256)
    aa("--epochs", type=int, default=40)
    aa("--momentum", type=float, default=0.9)
    aa("--rank_positive", type=int, default=10,
       help="this number of vectors are considered positives")
    aa("--rank_negative", type=int, default=50,
       help="these are considered negatives")

    group = parser.add_argument_group('Computation params')
    aa("--seed", type=int, default=1234)
    aa("--checkpoint_dir", type=str, default="",
       help="checkpoint directory")
    aa("--init_name", type=str, default="",
       help="checkpoint to load from")
    aa("--save_best_criterion", type=str, default="",
       help="for example r2=4,rank=10")
    aa("--quantizer_train", type=str, default="")
    aa("--lr_schedule", type=str, default="0.1,0.1,0.05,0.01")
    aa("--optim", type=str, default="sgd")
    aa("--device", choices=["cuda", "cpu", "auto"], default="auto")
    aa("--val_freq", type=int, default=10,
       help="frequency of validation calls")
    aa("--validation_quantizers", type=str, default="",
       help="r2 values to try in validation")
    aa("--print_results", type=int, default=0)


    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(args)

    name_for_c = "naive_triplet"
    results_file_name = "/home/shekhale/results/nns_graphs/" + args.database + "/train_results_" + name_for_c + ".txt"
    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n\n")
            rfile.write("START TRAINING \n")

    dint = args.dint
    dout = args.dout
    r_pos = args.rank_positive
    r_neg = args.rank_negative
    lam = args.lambda_uniform
    print ("load dataset %s" % args.database)
    (_, xb, xq, gt) = load_dataset(args.database, args.device, size=args.size_base, test=False, mnt=True)

    base_size = xb.shape[0]
    dim = xb.shape[1]
    threshold = int(base_size * 0.1)
    perm = np.random.permutation(base_size)
    xv = xb[perm[:threshold]]
    xt = xb

    print(xb[0][:10])

    xv = normalize_numpy(xv, args)
    xt = normalize_numpy(xt, args)
    xb = normalize_numpy(xb, args)
    xq = normalize_numpy(xq, args)

    print(xb[0][:10])

    print(xb.shape)
    print(xt.shape)
    print(xv.shape)


    xt = sanitize(xt)
    xv = sanitize(xv)
    xb = sanitize(xb)
    xq = sanitize(xq)


    print(xb[0][:10])

    print ("computing training ground truth")
    xt_gt = get_nearestneighbors_partly(xt, xt, r_pos, device=args.device, bs=10**5, needs_exact=True)
    # xt_gt = get_nearestneighbors(xt, xt, r_pos, device=args.device, needs_exact=True)
    # xt_gt = get_nearestneighbors(xb, xb, r_pos, device=args.device)

    print ("build network")
    net = nn.Sequential(
        nn.Linear(in_features=dim, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dout, bias=True),
        Normalize()
    )

    net.to(args.device)

    val_k = 2 * args.dout
    all_logs = triplet_optimize(xt, xv, xt_gt, xq, gt, net, args, lam, r_pos, r_neg, val_k, 0)

    yb = forward_pass(net, xb, 1024)

    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n")
            rfile.write(
                "Triplet, DATABASE %s, xt_size = %d, batch_size = %d, lat_dim = %d, k = %d, lam_u = %.5f, r_pos = %d, r_neg = %d , dint = %d, optim %s, margin = %.5f \n" %
                (args.database, xt.shape[0], args.batch_size, args.dout, val_k, lam, r_pos, r_neg, dint, args.optim, 0))
            # rfile.write("\n")
            log = all_logs[-1]
            rfile.write(
                "last perm = %.4f, train_top1_k = %.3f,  valid_top1_k = %.3f, query_top1_k = %.3f, query_top1_2k = %.3f \n" %
                (log['perm'], log['train_top1_k'], log['valid_top1_k'], log['query_top1_k'],
                 log['query_top1_2k']))

            rfile.write(
                "last logs: epochs %d, loss_uniform = %.6f, loss_triplet = %.6f, loss = %.6f, offending = %d, times %f %f %f \n" %
                (log['epoch'] + 1, log['loss_uniform'], log['loss_triplet'], log['loss'], log['offending'],
                 log['times'][0], log['times'][1], log['times'][2]))
            rfile.write("------------------------------------------------------ \n")

        k_nn = get_nearestneighbors_partly(xb, xb, 1000, args.device, bs=3*10**5, needs_exact=True)
        write_ivecs("/mnt/data/shekhale/models/nns_graphs/" + args.database + "/knn_1k.ivecs", k_nn)

        k_nn_latent = get_nearestneighbors_partly(yb, yb, 1000, args.device, bs=3*10**5, needs_exact=True)
        write_ivecs("/mnt/data/shekhale/models/nns_graphs/" + args.database + "/knn_lat_1k_" + name_for_c + ".ivecs", k_nn_latent)

        save_transformed_data(xb, net, args.database + "/" + args.database + "_base_" + name_for_c + ".fvecs", args.device)
        save_transformed_data(xq, net, args.database + "/" + args.database + "_query_" + name_for_c + ".fvecs", args.device)

