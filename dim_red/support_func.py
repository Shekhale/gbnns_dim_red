import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import itertools

from dim_red.data import write_fvecs, write_ivecs

try:
    import faiss
    hasfaiss = True
except:
    hasfaiss = False


def get_nearestneighbors_faiss(xq, xb, k, device, needs_exact=True, verbose=False):
    assert device in ["cpu", "cuda"]

    if verbose:
        print("Computing nearest neighbors (Faiss)")

    if needs_exact or device == 'cuda':
        index = faiss.IndexFlatL2(xq.shape[1])
    else:
        index = faiss.index_factory(xq.shape[1], "HNSW32")
        index.hnsw.efSearch = 64
    if device == 'cuda':
        index = faiss.index_cpu_to_all_gpus(index)

    start = time.time()
    index.add(xb)
    _, I = index.search(xq, k)
    if verbose:
        print("  NN search (%s) done in %.2f s" % (
            device, time.time() - start))

    return I


def cdist2(A, B):
    return  (A.pow(2).sum(1, keepdim = True)
             - 2 * torch.mm(A, B.t())
             + B.pow(2).sum(1, keepdim = True).t())


def top_dist(A, B, k):
    return cdist2(A, B).topk(k, dim=1, largest=False, sorted=True)[1]


def get_nearestneighbors_torch(xq, xb, k, device, needs_exact=False, verbose=False):
    if verbose:
        print("Computing nearest neighbors (torch)")

    assert device in ["cpu", "cuda"]
    start = time.time()
    xb, xq = torch.from_numpy(xb), torch.from_numpy(xq)
    xb, xq = xb.to(device), xq.to(device)
    bs = 500
    I = torch.cat([top_dist(xq[i*bs:(i+1)*bs], xb, k)
                   for i in range(xq.size(0) // bs)], dim=0)
    if verbose:
        print("  NN search done in %.2f s" % (time.time() - start))
    I = I.cpu()
    return I.numpy()


if hasfaiss:
    get_nearestneighbors = get_nearestneighbors_faiss
else:
    get_nearestneighbors = get_nearestneighbors_torch


def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')

# def save_ds(ds, net, args, file_name):
#
#     ds = torch.from_numpy(ds).to(args.device)
#     ds_lat = net(ds)
#     # print(ds.shape)
#
#     ds_lat = ds_lat.detach().numpy()
#     write_fvecs(file_name, ds_lat)


def get_histogram(x, num_steps, start, device):
    step = (1 - start) / (num_steps - 1)
    eps = (1 - start) /  num_steps
    # eps = (1 - start) / (2 * num_steps)
    t = torch.arange(start, 1 + step, step).view(-1, 1)
    tsize = t.size()[0]
    # t.cuda()
    t = t.to(device)
    # print(t)

    low_ind = (x < start).to(torch.bool)
    x[low_ind] = start
    # you can better
    if True:
        eq_ind = (x > 0.9999).to(torch.bool)
        x[eq_ind] = 0.9999
    # print(torch.sum(x))
    # x = x.reshape(-1)
    # y = x.reshape(-1)
    x_rep = x.reshape(-1).repeat(tsize, 1)
    # x_rep = y.repeat(tsize, 1)
    # print(x_rep.shape)

    x_rep_floor = (torch.floor((x_rep.data - start) / step) * step + start).float()
    x_rep_floor.to(device)
    # print(torch.sum(x_rep_floor))
    # print(x_rep_floor[0, :10])
    # t_step = torch.from_numpy(np.array(step)).to(device)
    # t_eps = torch.from_numpy(np.array(eps)).to(device)
    # eps.to(device)
    indsa = (x_rep_floor - (t - step) > -eps) & (x_rep_floor - (t - step) < eps)
    # indsa = ((x_rep_floor - (t - step) > -eps) & (x_rep_floor - (t - step) < eps)).to(torch.bool)

    # print(torch.sum(indsa))
    # indsa = (x_rep_floor - (t - t_step) > -t_eps) & (x_rep_floor - (t - t_step) < t_eps)
    # print(indsa.dtype)
    # zeros = torch.zeros((1, indsa.size()[1]))
    # zeros = torch.zeros((1, indsa.size()[1]), dtype=indsa.dtype, device=device)
    # if self.cuda:
    # zeros = zeros.to(device)
    # print(zeros.shape)
    # print(indsa.shape)
    indsbb = torch.cat((indsa, torch.zeros((1, indsa.size()[1]), dtype=indsa.dtype, device=device)), dim=0)
    indsb = indsbb[1:, :]
    # print(indsb.dtype)
    x_rep[~(indsb | indsa)] = 0

    # print(torch.sum(x_rep))
    # print(x_rep.dtype)
    # indsa corresponds to the first condition of the second equation of the paper
    x_rep[indsa] = (x_rep - t + step)[indsa] / step
    # indsb corresponds to the second condition of the second equation of the paper
    x_rep[indsb] = (-x_rep + t + step)[indsb] / step

    # print(torch.sum(x_rep))
    return x_rep.sum(1) / x_rep.shape[1]


def get_transform(xb, net, step, args):
    xb_lat = np.zeros((xb.shape[0], args.dout))
    net.eval()
    for i0 in range(0, xb.shape[0], step):
        i1 = min(i0 + step, xb.shape[0])
        xb_i = xb[i0:i1, :]
        xb_i = torch.from_numpy(xb_i).to(args.device)
        xb_l = net(xb_i.float())
        xb_lat[i0:i1, :] = xb_l.detach().cpu().numpy()
    xb_lat = sanitize(xb_lat)

    return xb_lat


def normalize_numpy(x, args):
    x_var = torch.from_numpy(x).to(args.device)
    x_var = x_var / x_var.norm(dim=-1, keepdim=True)
    x = x_var.detach().cpu().numpy()
    return x


def get_set_difference(pos, neg):
    if 2 * len(pos[0]) > len(neg[0]):
        print("Wrong pos and neg lenghts")
        return
    k_pos = len(pos[0])
    neg_dif = [0] * len(neg)
    for i in range(len(neg)):
        neg_dif[i] = list(set(neg[i]).difference(set(pos[i])))[:k_pos]

    return neg_dif


def calc_permutation(x, y, k):
    ans = 0
    for i in range(x.shape[0]):
        ans += len(list(set(x[i]) & set(y[i]))) / k
    return ans / x.shape[0]


def l2_dist(x, y):
    return np.sqrt(sum((x-y)**2))


def GetGroundTruth(X):
    th_nn = np.zeros(X.shape[0])
    pdist = l2_dist
    for i in range(X.shape[0]):
        if i != 0:
            nn = 0
            min_dist = pdist(X[i], X[0])
        else:
            nn = 1
            min_dist = pdist(X[i], X[1])
        for j in range(X.shape[0]):
            if i != j and pdist(X[i], X[j]) < min_dist:
                nn = j
                min_dist = pdist(X[i], X[j])
        th_nn[i] = nn
    return th_nn


def forward_pass(net, xall, bs=128, device=None):
    if device is None:
        device = next(net.parameters()).device
    xl_net = []
    net.eval()
    for i0 in range(0, xall.shape[0], bs):
        x = torch.from_numpy(xall[i0:i0 + bs])
        x = x.to(device)

        res = net(x)
        xl_net.append(res.data.cpu().numpy())

    return np.vstack(xl_net)


def forward_pass_enc(enc, xall, bs=128, device=None):
    if device is None:
        device = next(enc.parameters()).device
    xl_net = []
    enc.eval()
    for i0 in range(0, xall.shape[0], bs):
        x = torch.from_numpy(xall[i0:i0 + bs])
        x = x.to(device)
        res, _ = enc(x)
        xl_net.append(res.data.cpu().numpy())

    return np.vstack(xl_net)


def save_transformed_data(ds, model, path, device, enc=False):
    # ds = torch.from_numpy(ds).to(device)

    if enc:
        xb_var = torch.from_numpy(ds).to(device)
        xb_var = xb_var / xb_var.norm(dim=-1, keepdim=True)
        ds = xb_var.detach().cpu().numpy()
        del xb_var

    # ds = forward_pass_model(model, ds, 1024, lat=True)
    if enc:
        ds = forward_pass_enc(model, ds, 1024)
    else:
        ds = forward_pass(model, ds, 1024)
    # file_for_write_base = "data/" + path
    file_for_write_base = "/mnt/data/shekhale/data/" + path
    write_fvecs(file_for_write_base, ds)


def loss_permutation(x, y, args, k, size=10**4):
    perm = np.random.permutation(x.shape[0])
    k_nn_x = get_nearestneighbors(x[perm[:size]], x, k, args.device, needs_exact=True)
    k_nn_y = get_nearestneighbors(y[perm[:size]], y, k, args.device, needs_exact=True)
    perm_coeff = calc_permutation(k_nn_x, k_nn_y, k)
    print('top %d permutation is %.3f' % (k, perm_coeff))
    # with open("validation_sim.txt", "a") as rfile:
    #     rfile.write("epoch %d, perm = %.3f\n" % (int(epoch), perm_coeff))
    # logs['val'] = perm_coeff
    return perm_coeff


def loss_top_1_in_lat_top_k(xs, x, ys, y, args, kx, ky, size, name, fake_args=False):
    if xs.shape[0] != ys.shape[0]:
        print("wrong data")
    perm = np.random.permutation(xs.shape[0])
    top1_x = get_nearestneighbors(xs[perm[:size]], x, kx, args.device, needs_exact=True)
    top_neg_y = get_nearestneighbors(ys[perm[:size]], y, ky, args.device, needs_exact=True)
    ans_in_top_neg = 0
    for i in range(top1_x.shape[0]):
        if top1_x[i, -1] in top_neg_y[i]:
            ans_in_top_neg += 1
    print('%s: Part of top1_x in gt_lat_ %d = %.4f' % (name, ky, ans_in_top_neg / len(top1_x)))
    return ans_in_top_neg / top1_x.shape[0]


def loss_top_1_in_lat_top_k_new(gt, yb, yq, args, k):
    k_nn_yq = get_nearestneighbors(yq, yb, 2*k, args.device, needs_exact=True)
    ans_in_top_100 = 0
    ans_in_top_50 = 0
    for i in range(len(yq)):
        if gt[i, 0] in k_nn_yq[i]:
            ans_in_top_100 += 1
        if gt[i, 0] in k_nn_yq[i][:k]:
            ans_in_top_50 += 1
    print('QUERY: Part of gt in gt_lat_%d = %.4f' % (k, ans_in_top_50 / len(yq)))
    print('QUERY: Part of gt in gt_lat_%d = %.4f' % (2*k, ans_in_top_100 / len(yq)))

    return ans_in_top_50 / len(yq), ans_in_top_100 / len(yq)


def show_neighbours_distr(x, k, args, hist_steps=10, print_in_file=False, file_name=""):
    n = x.shape[0]
    knn = get_nearestneighbors(x, x, k, args.device, needs_exact=True)
    kth_neighbors_ind = knn[:, -1]
    distances = [0] * n
    for i in range(n):
        distances[i] = np.sqrt(np.sum((x[i] - x[kth_neighbors_ind[i]]) ** 2))
    dist_min = np.min(distances)
    dist_max = np.max(distances)
    delta = dist_max - dist_min + 0.000001
    hist = [0] * hist_steps
    for i in range(n):
        ind = int(hist_steps * (distances[i] - dist_min) / delta)
        hist[ind] += 1
    hist = [h / n for h in hist]
    # print(dist_min, dist_max)
    print("Histogram neig:", hist)

    hist_values = [0] * hist_steps
    for i in range(hist_steps):
        hist_values[i] = dist_min + i * delta / hist_steps
    if print_in_file:
        with open(file_name, "a") as rfile:
            rfile.write("Top %d neigbour distance distribution \n" %(k-1))
            # rfile.write(hist)
            rfile.write(" ".join(str(item)[:5] for item in hist) + "\n")
            rfile.write(" ".join(str(item)[:5] for item in hist_values) + "\n")
            # rfile.write(hist_values)

    return hist


def get_weights(x, k, args):
    n = x.shape[0]
    knn = get_nearestneighbors(x, x, k, args.device, needs_exact=True)
    kth_neighbors_ind = knn[:, -1]
    distances = [0] * n

    weights = [1 / d for d in distances]
    weights = weights / (sum(weights) / n)
    return weights


def calc_clasterization_coeff(graph, size=10**3, file_name=""):
    n, k = graph.shape
    distr = [0] * 10
    for i in range(size):
        inters = 0
        ind = random.randint(0, n - 1)
        for nb in graph[ind]:
            if nb != ind:
                inters += len(set(graph[ind]) & set(graph[nb])) # -2 because ind n for ind

        place = int(len(distr) * inters / (k * k)) # each edge two times
        distr[place] += 1 / size

    # print(distr)
    print("Claster distribution: " + " ".join(str(x)[:5] for x in distr))

    if file_name != "":
        with open(file_name, "a") as rfile:
            rfile.write("Claster distribution: " +" ".join(str(x)[:5] for x in distr))
            rfile.write("\n")

    return distr


def generate_uniform(n, d, nq, device, save=False):
    xb = torch.randn(size=(n, d)).to(device)
    xb = xb / xb.norm(dim=-1, keepdim=True)

    xq = torch.randn(size=(nq, d)).to(device)
    xq = xq / xq.norm(dim=-1, keepdim=True)

    x = xq.detach().cpu().numpy()
    xq = xq.detach().cpu().numpy()

    gt = get_nearestneighbors(xq, x, 100, device, needs_exact=True)
    if save:
        # path_start = "/uniform_"
        file_for_write_base = "/uniform_base_" + str(d) + ".fvecs"
        write_fvecs(file_for_write_base, xb)
        file_for_write_q = "/uniform_query_" + str(d) + ".fvecs"
        write_fvecs(file_for_write_q, xq)
        file_for_write_gt = "/uniform_gt_" + str(d) + ".ivecs"
        write_ivecs(file_for_write_gt, gt)

    return xb, xq, gt


def get_nearestneighbors_partly(xq, xb, k, device, bs=10**5, needs_exact=True, path=""):

    knn = []

    for i0 in range(0, xq.shape[0], bs):
        xq_p = xq[i0:i0 + bs]
        res = get_nearestneighbors(xq_p, xb, k, device, needs_exact)
        knn.append(res)
    if path != "":
        write_ivecs(path, np.vstack(knn))
    return np.vstack(knn)


def get_nearestneighbors_brute(xq, xb, device, bs=10**6, needs_exact=True, sort=False, bucket=100, size = 100):
    knn = []
    for i in range(0, xq.shape[0], bs):
        xq_p = xq[i:i + bs]
        knn_i = []
        for j in range(0, xb.shape[0], bs):
            print(i, j)
            xb_p = xb[j:j + bs]
            res = get_nearestneighbors(xq_p, xb_p, bucket , device, needs_exact)
            res += j
            res = np.array(res, dtype='int32')
            knn_i.append(res)
        knn.append(np.hstack(knn_i))
    knn = np.vstack(knn)
    if sort:
        # size = 100
        knn_sorted = np.zeros((knn.shape[0], size))
        for i in range(knn.shape[0]):
            # print(i)
            neighborhood = []
            for j in knn[i]:
                neighborhood.append((l2_dist(xq[i], xb[j]), j))
            neighborhood.sort()
            knn_sorted[i] = list(list(zip(*neighborhood))[1])[:size]
            # knn[i] = [pair[1] for pair in neighborhood]
        return np.array(knn_sorted, dtype='int32')
    return knn


def PrintAnsQueryStat(qftt, aftt, x, size=10**4):
    offending = 0
    perm = np.random.permutation(x.shape[0])
    one_print = True
    for i in range(size):
        ind = perm[i]
        query = qftt[ind]
        answer = aftt[ind]
        if one_print:
            print(query.shape, len(query))
            print(answer.shape)
            one_print = False
        for j in range(len(query)):
            if l2_dist(x[query[j]], x[answer[j]]) > l2_dist(x[query[j]], x[ind]):
                offending += 1

    print("Offending ", offending, ", ", offending / size)


def GetIntrinsicDimension(x, knn, calc_size=10**4):
    edges = [10, 15, 20]
    dims = []
    perm = np.random.permutation(x.shape[0])
    problems = 0
    for edge in edges:
        d_low = 0
        for i in perm[:calc_size]:
            d_low_cur = 0
            dist_k = l2_dist(x[i], x[knn[i][edge]])
            for j in range(1, edge):
                dist_j = l2_dist(x[i], x[knn[i][j]])
                if dist_j > 0.0001:
                    d_low_cur += np.log(dist_k / dist_j)
            if d_low_cur > 0:
                d_low_cur = 1 / d_low_cur
                d_low_cur *= edge - 1
                d_low += d_low_cur
            else:
                problems += 1

        d_low /= calc_size
        dims.append(d_low)

    d_low = np.mean(dims)
    print(dims)
    print("Intrinsic dimension is  %.3f" % (d_low))
    print("problems  %d" % (problems))


def get_knn_distr(x, knn, k, filename, calc_size=10**4):
    perm = np.random.permutation(x.shape[0])
    for i in perm[:calc_size]:
        dist_k = l2_dist(x[i], x[knn[i][k]])
        if dist_k > 0:
            with open(filename, 'a') as out:
                out.write(str(dist_k)[:6] + ' ')

    with open(filename, 'a') as out:
        out.write('\n')


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2., dim=1)


def save_picture(data, y, args, k, lambda_uniform, alpha1=0, alpha2=0, numb=1):
    c1 = np.array([[np.cos(alpha1), np.sin(alpha1)],[ -np.sin(alpha1),  np.cos(alpha1)]])
    cols = [0,1]
    data[:, cols] = data[:, cols].dot(c1)

    c2 = np.array([[np.cos(alpha2), np.sin(alpha2)],[ -np.sin(alpha2),  np.cos(alpha2)]])
    cols = [0, 2]
    data[:, cols] = data[:, cols].dot(c2)
    # print(data1.shape)
    fig, ax = plt.subplots()
    # ax = fig.gca(projection='3d')
    # ax.scatter(*latents.T, c=y, s=2)
    plt.scatter(*data.T, c=y)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    #smth new
    # fig.savefig("pictures/mnist_catalyst_k_pos_" + str(r_pos) + "_k_neg_" + str(r_neg) + "_lam_" + str(lam) + ".png")
    fig.savefig("mnist_tsne/mnist_tsne_k_nn_" + str(k) + "_" + str(lambda_uniform) + "_" + str(numb) + ".png")


def repeat(l, r):
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))


def pairwise_NNs_inner(x):
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n + 1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I
