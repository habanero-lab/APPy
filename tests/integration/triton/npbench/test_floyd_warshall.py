# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.int32):
    path = np.fromfunction(lambda i, j: i * j % 7 + 1, (N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = 999

    return path


def kernel_np(path):
    for k in range(path.shape[0]):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
    return path


import appy


@appy.jit(backend="triton", dump_code=True)
def kernel_appy(path):
    for k in range(path.shape[0]):
        #pragma parallel for          
        for i in range(path.shape[0]):
            #pragma simd
            for j in range(path.shape[1]):
                path[i, j] = np.minimum(path[i, j], path[i, k] + path[k, j])
    return path


def test():
    N = 100

    path = initialize(N)
    path_ref = kernel_np(path)
    path = kernel_appy(path)
    assert np.allclose(path, path_ref, atol=1e-6)