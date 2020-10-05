from __future__ import division
import argparse
import numpy as np
import torch

from dim_red.triplet import train_triplet
from dim_red.angular import train_angular
from wrap.triplet_wrap import train_triplet as train_triplet_wrap

from dim_red.support_func import  sanitize
from dim_red.data import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('dataset options')
    aa("--database", default="sift")
    aa("--method", type=str, default="triplet")

    group = parser.add_argument_group('Model hyperparameters')
    aa("--dout", type=int, default=16,
       help="output dimension")
    aa("--dint", type=int, default=1024)
    group = parser.add_argument_group('Computation params')
    aa("--seed", type=int, default=1234)
    aa("--device", choices=["cuda", "cpu", "auto"], default="auto")
    aa("--val_freq", type=int, default=10,
       help="frequency of validation calls")
    aa("--optim", type=str, default="sgd")
    aa("--print_results", type=int, default=0)
    aa("--save", type=int, default=0)
    aa("--full", type=int, default=0)
    aa("--val_freq_search", type=int, default=5,
       help="frequency of validation calls")
    aa("--save_knn_1k", type=int, default=0)
    aa("--save_optimal", type=int, default=0)
    aa("--batch_size", type=int, default=64)
    aa("--epochs", type=int, default=40)
    aa("--lr_schedule", type=str, default="0.1,0.1,0.05,0.01")
    aa("--momentum", type=float, default=0.9)

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(args)

    results_file_name = "/home/shekhale/results/dim_red/" + args.database + "/train_results_" + args.method + ".txt"
    if args.print_results > 0:
        with open(results_file_name, "a") as rfile:
            rfile.write("\n\n")
            rfile.write("START TRAINING \n")

    print ("load dataset %s" % args.database)
    (_, xb, xq, _) = load_dataset(args.database, args.device, calc_gt=False, mnt=True)

    base_size = xb.shape[0]
    threshold = int(base_size * 0.01)
    perm = np.random.permutation(base_size)
    xv = xb[perm[:threshold]]
    if args.full:
        xt = xb
    else:
        xt = xb[perm[threshold:]]

    print(xb.shape, xt.shape, xv.shape, xq.shape)

    xt = sanitize(xt)
    xv = sanitize(xv)
    xb = sanitize(xb)
    xq = sanitize(xq)

    if args.method == "triplet":
        train_triplet(xb, xt, xv, xq, args, results_file_name)
    elif args.method == "angular":
        train_angular(xb, xt, xv, xq, args, results_file_name, perm)
    else:
        print("Select an available method")