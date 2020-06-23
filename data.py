from os.path import join
import numpy as np
from support_func import get_nearestneighbors, sanitize


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
        start = ""
    else:
        start = ""
    paths = {
        "sift": start + "data/sift/",
        "gist": start + "data/gist/",
        "glove": start + "data/glove/",
        "deep": start + "data/deep/",
        "uniform_low": start + "data/synthetic/"
    }

    return paths[s]


def load_sift(device, size = 10 ** 6, test=True, mnt=False):
    basedir = getBasedir("sift", mnt)

    # dbsize = int(size / 10 ** 6)

    xb = mmap_fvecs(join(basedir, 'sift_base.fvecs'))
    xq = mmap_fvecs(join(basedir, 'sift_query.fvecs'))
    # trim xb to correct size
    # xb = xb[:dbsize * 1000 * 1000]
    gt = ivecs_read(join(basedir, 'sift_groundtruth.ivecs'))
    # xt = mmap_fvecs(join(basedir, 'sift_learn.fvecs'))
    xt = mmap_fvecs(join(basedir, 'sift_base.fvecs'))
    # else:
    #     xb = xt[:size]
    #     xq = xt[size:size+qsize]
    #     xt = xt[size+qsize:]

    xb, xq = sanitize(xb), sanitize(xq)
    if not test:
        gt = get_nearestneighbors(xq, xb, 100, device)

    return xt, xb, xq, gt


def load_gist(device, size = 10 ** 6, test=True, mnt=False):
    basedir = getBasedir("gist", mnt)

    # dbsize = int(size / 10 ** 6)

    xb = mmap_fvecs(join(basedir, 'gist_base.fvecs'))
    xq = mmap_fvecs(join(basedir, 'gist_query.fvecs'))
    gt = ivecs_read(join(basedir, 'gist_groundtruth.ivecs'))

    xb, xq = sanitize(xb), sanitize(xq)
    if not test:
        gt = get_nearestneighbors(xq, xb, 100, device, needs_exact=True)

    return xb, xb, xq, gt


def load_deep(device, size = 10 ** 6, test=True, mnt=False):
    basedir = getBasedir("deep", mnt)

    # dbsize = int(size / 10 ** 6)

    xb = mmap_fvecs(join(basedir, 'deep_base.fvecs'))
    xq = mmap_fvecs(join(basedir, 'deep_query.fvecs'))
    gt = ivecs_read(join(basedir, 'deep_groundtruth.ivecs'))

    xb, xq = sanitize(xb), sanitize(xq)
    if not test:
        gt = get_nearestneighbors(xq, xb, 100, device, needs_exact=True)

    return xb, xb, xq, gt


def load_glove(device, size = 10 ** 6, test=True, mnt=False):
    basedir = getBasedir("glove", mnt)

    # dbsize = int(size / 10 ** 6)

    xb = mmap_fvecs(join(basedir, 'glove_base.fvecs'))
    xq = mmap_fvecs(join(basedir, 'glove_query.fvecs'))
    gt = ivecs_read(join(basedir, 'glove_groundtruth.ivecs'))

    xb, xq = sanitize(xb), sanitize(xq)
    if not test:
        gt = get_nearestneighbors(xq, xb, 100, device, needs_exact=True)

    return xb, xb, xq, gt


def load_dataset(name, device, size=10**6, test=True, mnt=False):
    if name == "sift":
        return load_sift(device, size, test, mnt)
    elif name == "gist":
        return load_gist(device, size, test, mnt)
    elif name == "deep":
        return load_deep(device, size, test, mnt)
    elif name == "glove":
        return load_glove(device, size, test, mnt)


