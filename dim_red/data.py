from os.path import join
# from support_func import get_nearestneighbors, sanitize
import numpy as np
from struct import pack, unpack
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


def read_ivecs(filename, max_size=None):
    max_size = max_size or float('inf')
    with open(filename, "rb") as f:
        vecs = []
        while True:
            header = f.read(4)
            if not len(header): break
            dim, = unpack('<i', header)
            vec = unpack('i' * dim, f.read(4 * dim))
            vecs.append(vec)
            if len(vecs) >= max_size: break
    maxl = 0
    for vec in vecs:
        maxl = max(maxl, len(vec))
    # print(maxl)
    ans = np.zeros((len(vecs), maxl), dtype="int32")
    for i in range(len(vecs)):
        for j in range(len(vecs[i])):
            ans[i][j] = vecs[i][j]

    return ans


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


