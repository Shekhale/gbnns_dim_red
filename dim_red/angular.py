from __future__ import division
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch
import itertools

from dim_red.support_func import  forward_pass, Normalize, stopping_time,\
    repeat, get_nearestneighbors_partly, save_transformed_data, ifelse,\
    validation_function, save_net_as_matrix

from dim_red.data import write_fvecs

net_style = "angular"


def get_graph_dot_prod(graph_cur, xt_var):
    graph_directions = graph_cur - xt_var.reshape(graph_cur.shape[0], 1, graph_cur.shape[-1])
    graph_directions = graph_directions / (graph_directions.norm(dim=-1, keepdim=True) + 0.00001)
    dot_prod = torch.bmm(graph_directions, graph_directions.permute(0, 2, 1))

    return dot_prod


def angular_loss(graph, xt_var, net, bn, args):
    x_dot_prod = get_graph_dot_prod(graph, xt_var)

    graph_cur_y = net(graph.reshape(-1, graph.shape[-1])).reshape(bn, -1, args.dout)
    yt_var_cur = net(xt_var)
    y_dot_prod = get_graph_dot_prod(graph_cur_y, yt_var_cur)

    return ((x_dot_prod - y_dot_prod) ** 2).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def angular_optimize(xt, xv, xq, net, args, lambda_uniform, lambda_triplet, lambda_ang, val_k, graph, knn, k_pos, k_neg,
                     valid, dfl):
    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    n = xt.shape[0]
    xt_var = torch.from_numpy(xt).to(args.device)
    acc =[]
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
        rank_pos = np.random.choice(k_pos, size=n)
        positive_idx = knn[np.arange(n), rank_pos]
        # positive_idx = gt_nn[np.arange(n), rank_pos]

        # Sample negatives for triplet
        net.eval()
        print("  Forward pass")
        xl_net = forward_pass(net, xt, 1024)
        print("  Distances")
        I = get_nearestneighbors_partly(xl_net, xl_net, k_neg, args.device, bs=3*10**5, needs_exact=False)
        negative_idx = I[:, -1]

        # training pass
        print("  Train")
        net.train()
        avg_ang, avg_uniform, avg_loss = 0, 0, 0
        offending = idx_batch = 0
        avg_triplet = 0

        # process dataset in a random order
        perm = np.random.permutation(n)

        t1 = time.time()

        for i0 in range(0, n, args.batch_size):
            i1 = min(i0 + args.batch_size, n)
            bn = i1 - i0
            data_idx = perm[i0:i1]

            # anchor, positives, negatives
            xt_cur = xt_var[data_idx]
            pos = xt_var[positive_idx[data_idx]]
            neg = xt_var[negative_idx[data_idx]]

            # do the forward pass (+ record gradients)
            yt_cur = net(xt_cur)
            pos, neg = net(pos), net(neg)

            # triplet loss
            per_point_loss = pdist(yt_cur, pos) - pdist(yt_cur, neg)
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()

            # # entropy loss
            # I = pairwise_NNs_inner(ins.data)
            # distances = pdist(ins, ins[I])
            # loss_uniform_2 = - torch.log(distances).mean()

            # angular loss
            graph_cur = xt_var[graph[data_idx]]
            loss_ang = angular_loss(graph_cur,  xt_var[data_idx], net, bn, args)

            # uniform
            loss_uniform = uniform_loss(yt_cur)

            # combined loss
            loss = lambda_ang * loss_ang + lambda_uniform * loss_uniform + lambda_triplet * loss_triplet

            # collect some stats
            avg_ang += loss_ang.data.item()
            avg_triplet += loss_triplet.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_batch += 1

        avg_ang /= idx_batch
        avg_uniform /= idx_batch
        avg_triplet /= idx_batch
        avg_loss /= idx_batch

        logs = {
            'epoch': epoch,
            'loss_ang': avg_ang,
            'loss_uniform': avg_uniform,
            'loss_triplet': avg_triplet,
            'loss': avg_loss,
            'offending': offending,
            'lr': args.lr
        }

        t2 = time.time()

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            logs_val = validation_function(net, xt, xv, xq, args, val_k)
            logs.update(logs_val)

        t3 = time.time()

        print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' lr = %f'
               ' loss = %g = %g + lam_u * %g + lam_tr * %g, offending %d' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            args.lr,
            avg_loss, avg_ang, avg_uniform, avg_triplet, offending
        ))

        logs['times'] = (t1 - t0, t2 - t1, t3 - t2)
        all_logs.append(logs)

        if args.val_freq_search > 0 and ((epoch + 1) % args.val_freq_search == 0 or epoch == args.epochs - 1):
            import wrap.c_support as c_support
            net.eval()
            dim = xt.shape[1]
            yt = forward_pass(net, xt, 1024)
            knn_low_path = "/mnt/data/shekhale/models/nns_graphs/" + args.database + "/knn_1k_" + net_style + valid + ".ivecs"
            get_nearestneighbors_partly(yt, yt, 1000, args.device, bs=3 * 10 ** 5, needs_exact=True, path=knn_low_path)
            save_transformed_data(xt, net, args.database + "/" + args.database + "_base_" + net_style + valid + ".fvecs",
                                  args.device)
            if len(valid) > 0:
                save_transformed_data(xv, net, args.database + "/" + args.database + "_query_" + net_style + valid + ".fvecs",
                                      args.device)
            else:
                save_transformed_data(xq, net, args.database + "/" + args.database + "_query_" + net_style + ".fvecs",
                                      args.device)
            acc_cur = c_support.get_graphs_and_search_tests("a", dfl, dim, args.dout, xq.shape[0], valid_char,
                                                            xt.shape[0], 0, False)
            acc_cur = round(acc_cur, 5)
            if args.save_optimal > 0:
                if len(acc) > 0 and acc_cur > max(acc):
                    net_path = "/mnt/data/shekhale/models/nns_graphs/" + str(args.database) + "/" + \
                               str(args.database) + "_net_angular_optimal.pth"
                    torch.save(net.state_dict(), net_path)
            acc.append(acc_cur)
            net.train()
            print("Acc list ", acc)
            logs['acc'] = acc
            # if stopping_time(acc, 0.001):
            #     if (epoch + 1) % args.val_freq != 0 and epoch != args.epochs - 1:
            #         logs_val = ValidationFunction(net, xt, xv, xq, args, val_k)
            #         logs.update(logs_val)
            #     return all_logs

    return all_logs


def train_angular(xb, xt, xv, xq, args, results_file_name, perm):
    dints = ifelse(args.dint, [512, 128, 64, 32])
    lambdas_triplet = [1]
    lambdas_ang = [1]
    lambdas_uni = [0.01]
    random_graph_sizes = [25]
    ranks_pos = [5]
    ranks_neg = [10]
    dataset_first_letter = args.database[0]
    if args.database == "sift":
        ranks_pos = [5]
        ranks_neg = [20]
        lambdas_uni = [0.01]
    elif args.database == "glove":
        ranks_pos = [5]
        ranks_neg = [40]
        lambdas_ang = [1]
        lambdas_uni = [0.1]
        dataset_first_letter = "w"

    learning_params = list(itertools.product(lambdas_uni, lambdas_triplet, lambdas_ang, dints, ranks_pos, ranks_neg,
                                             random_graph_sizes))
    print(learning_params)

    for lambda_uniform, lambda_triplet, lambda_ang, dint, k_pos, k_neg, rgs in learning_params:
        print(lambda_uniform, lambda_triplet, lambda_ang, dint, k_pos, k_neg, rgs)
        gt_xt = get_nearestneighbors_partly(xt, xt, k_pos, device=args.device, bs=10**6, needs_exact=True)
        graph_random = np.random.randint(0, xt.shape[0], (xt.shape[0], rgs))
        dim = xt.shape[1]
        dout = args.dout

        print("build network")

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

        all_logs = angular_optimize(xt, xv, xq, net, args, lambda_uniform, lambda_triplet, lambda_ang, val_k,
                                    graph_random, gt_xt, k_pos, k_neg, valid, dataset_first_letter)


# -------------------------------------------------------------------------------------------------------------------

        if args.print_results > 0:
            with open(results_file_name, "a") as rfile:
                rfile.write("\n")

                rfile.write(
                    "Angular, DATABASE %s, xt_size = %d, batch_size = %d, lat_dim = %d, k = %d, net_size = %d  \n" %
                    (args.database, xt.shape[0], args.batch_size, args.dout, val_k, len(list(net.state_dict().keys()))))
                rfile.write(
                    "lam_u = %.7f, lam_tr = %.7f, lam_ang = %.3f, r_pos = %d, r_neg = %d, dint = %d, rgs = %s \n" %
                    (lambda_uniform, lambda_triplet, lambda_ang, k_pos, k_neg, dint, rgs))

                log = all_logs[-1]
                rfile.write(
                    "last perm = %.4f, train_top1_k = %.3f,  valid_top1_k = %.3f, query_top1_k = %.3f, query_top1_2k = %.3f \n" %
                    (log['perm'], log['train_top1_k'], log['valid_top1_k'], log['query_top1_k'],
                     log['query_top1_2k']))

                rfile.write(
                    "last logs: epochs %d, loss_uniform = %.6f, loss_triplet = %.6f, loss_ang = %.6f, loss = %.6f, offending = %d, times %f %f %f \n" %
                    (log['epoch'] + 1, log['loss_uniform'],  log['loss_triplet'], log['loss_ang'], log['loss'], log['offending'],
                     log['times'][0], log['times'][1], log['times'][2]))
                if args.val_freq_search > 0:
                    rfile.write("Acc list: ")
                    rfile.write(' '.join([str(e) for e in log['acc']]))
                rfile.write("\n")
                rfile.write("------------------------------------------------------ \n")

        if args.save > 0:
            yb = forward_pass(net, xb, 1024)
            yq = forward_pass(net, xq, 1024)
            if args.save_knn_1k > 0:
                knn_low_path = models_path + "_knn_1k_" + net_style + ".ivecs"
                get_nearestneighbors_partly(yb, yb, 1000, args.device, bs=3*10**5, needs_exact=True, path=knn_low_path)

            gt_low_path = "/mnt/data/shekhale/data/" + args.database + "/" + args.database + "_groundtruth_" + net_style + ".ivecs"
            get_nearestneighbors_partly(yq, yb, 100, args.device, bs=3*10**5, needs_exact=True, path=gt_low_path)

            save_transformed_data(xb, net, args.database + "/" + args.database + "_base_" + net_style + ".fvecs", args.device)
            save_transformed_data(xq, net, args.database + "/" + args.database + "_query_" + net_style + ".fvecs", args.device)

# -------------------------------------------------------------------------------------------------------------------

        params_string = str(dout) + "_l_" + str(int(-np.log10(lambda_uniform))) + "_1m_" + str(k_pos) + "_" + \
            str(k_neg) + "_w_" + str(dint) + "_e_" + str(args.epochs)
        net_path = models_path + "_net_" + params_string + ".pth"
        net_script_path = models_path + "_net_" + params_string + "_scr.pth"

        if args.save > 0:
            torch.save(net.state_dict(), net_path)
            net_script = torch.jit.script(net)
            net_script.save(net_script_path)
            save_net_as_matrix(net, models_path + "_net_as_matrix_" + args.method + "_optimal")
