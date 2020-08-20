from os.path import join
# from support_func import get_nearestneighbors, sanitize
import numpy as np
from struct import pack


def write_fvecs(filename, vecs):
    with open(filename, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('f' * dim, *list(vec)))


def write_ivecs(filename, vecs):
    with open(filename, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('i' * dim, *list(vec)))


def write_edges_dict(filename, edges):
    with open(filename, "wb") as f:
        for from_vertex_id, to_vertex_ids in edges.items():
            dim = len(to_vertex_ids)
            f.write(pack('<i', dim))
            f.write(pack('i' * dim, *list(to_vertex_ids)))


def write_edges_list(filename, edges):
    with open(filename, "wb") as f:
        for to_vertex_ids in edges:
            dim = len(to_vertex_ids)
            f.write(pack('<i', dim))
            f.write(pack('i' * dim, *list(to_vertex_ids)))


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def getBasedir(s, mnt=False):
    if mnt:
        start = "/mnt/data/shekhale/"
    else:
        start = "/home/shekhale/"
    paths = {
        "sift": start + "data/sift/sift",
        "gist": start + "data/gist/gist",
        "glove": start + "data/glove/glove",
        "deep": start + "data/deep/deep",
        "boltalka_dssm": start + "data/boltalka_dssm/boltalka_dssm",
        "kcd_visual": start + "data/kcd_visual/kcd_visual",
        "kcd_t2t": start + "data/kcd_t2t/kcd_t2t",
        "uniform_low": start + "data/synthetic/"
    }

    return paths[s]


def load_simple(device, database, calc_gt=False, mnt=False):
    basedir = getBasedir(database, mnt)
    xb = mmap_fvecs(basedir + '_base.fvecs')
    xq = mmap_fvecs(basedir + '_query.fvecs')
    gt = ivecs_read(basedir + '_groundtruth.ivecs')

    xb, xq = np.ascontiguousarray(xb), np.ascontiguousarray(xq)
    # if calc_gt:
    #     gt = get_nearestneighbors(xq, xb, 100, device, needs_exact=True)

    # return xb, xb, xq, xq
    return xb, xb, xq, gt


def load_dataset(name, device, size=10**6, calc_gt=False, mnt=True):
    if name == "sift":
        return load_simple(device, "sift", calc_gt, mnt)
    elif name == "gist":
        return load_simple(device, "gist", calc_gt, mnt)
    elif name == "deep":
        return load_simple(device, "deep", calc_gt, mnt)
    elif name == "glove":
        return load_simple(device, "glove", calc_gt, mnt)
    elif name == "boltalka_dssm":
        return load_simple(device, "boltalka_dssm", calc_gt, mnt)
    elif name == "kcd_t2t":
        return load_simple(device, "kcd_t2t", calc_gt, mnt)
    elif name == "kcd_visual":
        return load_simple(device, "kcd_visual", calc_gt, mnt)


