from __future__ import division
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch
import itertools


from dim_red.support_func import  loss_permutation, loss_top_1_in_lat_top_k, normalize_numpy,\
    get_nearestneighbors, sanitize, forward_pass, Normalize, stopping_time,\
    repeat, pairwise_NNs_inner, get_nearestneighbors_partly, save_transformed_data, ifelse, validation_function,\
    save_net_as_matrix

from dim_red.data import write_fvecs, write_ivecs

net_style = "triplet"


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def triplet_optimize(xt, xv, gt_nn, xq, net, args, lambda_uniform, k_pos, k_neg, val_k, margin, dfl, valid):
    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    N = gt_nn.shape[0]
    acc = []
    xt_var = torch.from_numpy(xt).to(args.device)
    valid_char = ""
    if len(valid) > 0:
        valid_char = "v"

    # prepare optimizer
    optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    pdist = nn.PairwiseDistance(2)

    all_logs = []
    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        t0 = time.time()

        # Sample positives for triplet
        rank_pos = np.random.choice(k_pos, size=N)
        positive_idx = gt_nn[np.arange(N), rank_pos]

        # Sample negatives for triplet
        net.eval()
        print("  Forward pass")
        xl_net = forward_pass(net, xt, 1024)
        print("  Distances")
        I = get_nearestneighbors(xl_net, xl_net, k_neg, args.device, needs_exact=False)
        negative_idx = I[:, -1]

        # training pass
        print("  Train")
        net.train()
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

            # uniform
            loss_uniform = uniform_loss(ins)

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

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            logs_val = validation_function(net, xt, xv, xq, args, val_k)
            logs.update(logs_val)
            net.train()

        t3 = time.time()

        print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' lr = %f'
               ' loss = %g = %g + lam * %g, offending %d' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            args.lr,
            avg_loss, avg_triplet, avg_uniform, offending
        ))

        logs['times'] = (t1 - t0, t2 - t1, t3 - t2)

        if args.val_freq_search > 0 and ((epoch + 1) % args.val_freq_search == 0 or epoch == args.epochs - 1):
            import wrap.c_support as c_support
            net.eval()
            dim = xt.shape[1]
            yt = forward_pass(net, xt, 1024)
            knn_low_path = "/mnt/data/shekhale/models/nns_graphs/" + args.database + "/knn_1k_" + net_style + valid + ".ivecs"
            get_nearestneighbors_partly(yt, yt, 1000, args.device, bs=3 * 10 ** 5, needs_exact=True, path=knn_low_path)
            save_transformed_data(xt, net, args.database + "/" + args.database + "_base_" + net_style + valid + ".fvecs",
                                  args.device)
            save_transformed_data(xv, net, args.database + "/" + args.database + "_query_" + net_style + valid + ".fvecs",
                                  args.device)
            acc_cur = c_support.get_graphs_and_search_tests("t", dfl, dim, args.dout, xq.shape[0], valid_char, xt.shape[0], False)
            acc_cur = round(acc_cur, 5)
            if args.save_optimal > 0:
                if len(acc) > 0 and acc_cur > max(acc):
                    net_path = "/mnt/data/shekhale/models/nns_graphs/" + str(args.database) + "/" + \
                               str(args.database) + "_net_triplet_optimal.pth"
                    torch.save(net.state_dict(), net_path)
            acc.append(acc_cur)
            net.train()
            print("Acc list ", acc)
            logs['acc'] = acc
            if stopping_time(acc, 0.002):
                return all_logs

    return all_logs


def train_triplet(xb, xt, xv, xq, args, results_file_name):

    lambdas = ifelse(args.lambda_uniform, list(np.logspace(-2, 0, 3)))
    dints = ifelse(args.dint, [512])
    ranks_pos = [5]
    ranks_neg = [10]
    dataset_first_letter = args.database[0]
    if args.database == "sift":
        ranks_pos = [5]
        ranks_neg = [10]
        lambdas = [0.01]
    elif args.database == "glove":
        ranks_pos = [5]
        ranks_neg = [40]
        lambdas = [0.01]
        dataset_first_letter = "w"

    learning_params = list(itertools.product(lambdas, dints, ranks_pos, ranks_neg))

    print(learning_params)

    for lambda_uniform, dint, k_pos, k_neg in learning_params:
        margin = 0
        print(lambda_uniform, dint, k_pos, k_neg)

        dim = xt.shape[1]
        dout = args.dout

        print ("computing training ground truth")
        xt_gt = get_nearestneighbors(xt, xt, k_pos, device=args.device, needs_exact=True)

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

        models_path =  "/mnt/data/shekhale/models/nns_graphs/" + args.database + "/" + args.database
        data_path =  "/mnt/data/shekhale/data/" + args.database + "/" + args.database
        valid = ""
        if args.full != 1:
            valid = "_valid"
            base_path = data_path + "_base_valid.fvecs"
            write_fvecs(base_path, xt)
            query_path = data_path + "_query_valid.fvecs"
            write_fvecs(query_path, xv)
            gt_path = data_path +  "_groundtruth_valid.ivecs"
            get_nearestneighbors_partly(xv, xt, 100, args.device, bs=3 * 10 ** 5, needs_exact=True, path=gt_path)

        all_logs = triplet_optimize(xt, xv, xt_gt, xq, net, args, lambda_uniform, k_pos, k_neg, val_k, margin,
                                    dataset_first_letter, valid)

        if args.print_results > 0:
            with open(results_file_name, "a") as rfile:
                rfile.write("\n")

                rfile.write(
                    "Triplet, DATABASE %s, xt_size = %d, batch_size = %d, lat_dim = %d\n" %
                    (args.database, xt.shape[0], args.batch_size, args.dout))
                rfile.write(
                    "k = %d, lam_u = %.7f, r_pos = %d, r_neg = %d , dint = %d, margin = %.5f,"
                    " net_state_dict_size = %d \n" %
                    (val_k, lambda_uniform, k_pos, k_neg, dint, margin, len(list(net.state_dict().keys()))))

                log = all_logs[-1]
                rfile.write(
                    "last perm = %.4f, train_top1_k = %.3f,  valid_top1_k = %.3f, query_top1_k = %.3f,"
                    " query_top1_2k = %.3f \n" %
                    (log['perm'], log['train_top1_k'], log['valid_top1_k'], log['query_top1_k'],
                     log['query_top1_2k']))

                rfile.write(
                    "last logs: epochs %d, loss_uniform = %.6f, loss_triplet = %.6f, loss = %.6f, offending = %d,"
                    " times %f %f %f \n" %
                    (log['epoch'] + 1, log['loss_uniform'], log['loss_triplet'], log['loss'], log['offending'],
                     log['times'][0], log['times'][1], log['times'][2]))
                if args.val_freq_search > 0:
                    rfile.write("Acc list: ")
                    rfile.write(' '.join([str(e) for e in log['acc']]))

                rfile.write("\n")
                rfile.write("------------------------------------------------------ \n")

        yb = forward_pass(net, xb, 1024)
        yq = forward_pass(net, xq, 1024)
        if args.save > 0:
            if args.save_knn_1k > 0:
                knn_low_path = models_path + "_knn_1k_" + net_style + ".ivecs"
                get_nearestneighbors_partly(yb, yb, 1000, args.device, bs=3*10**5, needs_exact=True, path=knn_low_path)

            gt_low_path = "/mnt/data/shekhale/data/" + args.database + "/"\
                          + args.database + "_groundtruth_" + net_style + ".ivecs"
            get_nearestneighbors_partly(yq, yb, 100, args.device, bs=3*10**5, needs_exact=True, path=gt_low_path)

            save_transformed_data(xb, net, args.database + "/" + args.database + "_base_" + net_style + ".fvecs",
                                  args.device)
            save_transformed_data(xq, net, args.database + "/" + args.database + "_query_" + net_style + ".fvecs",
                                  args.device)

# -------------------------  SAVING PART  --------------------------------------------------------------------

        params_string = str(dout) + "_l_" + str(int(-np.log10(lambda_uniform))) + "_1m_" + str(k_pos) + "_" + \
            str(k_neg) + "_w_" + str(dint) + "_e_" + str(args.epochs)
        net_path = models_path + "_net_" + params_string + ".pth"
        net_script_path = models_path + "_net_" + params_string + "_scr.pth"

        if args.save > 0:
            torch.save(net.state_dict(), net_path)
            net_script = torch.jit.script(net)
            net_script.save(net_script_path)
            save_net_as_matrix(net, models_path + "_net_as_matrix_" + args.method + "_optimal")
