# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import appy


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


@appy.jit(backend="cuda", dump_code=False)
def kernel_appy(path):
    for k in range(path.shape[0]):
        #pragma parallel for
        for i in range(path.shape[0]):
            #pragma simd
            for j in range(path.shape[1]):
                path[i, j] = min(path[i, j], path[i, k] + path[k, j])
    return path


@appy.jit(backend="cuda", dump_code=False)
def kernel_appy2(path):
    for k in range(path.shape[0]):
        #pragma parallel for
        for i in range(path.shape[0]):
            path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])
    return path


def test():
    N = 100

    path1 = initialize(N)
    path2 = initialize(N).copy()
    path3 = initialize(N).copy()

    kernel_np(path1)
    kernel_appy(path2)
    kernel_appy2(path3)

    assert np.allclose(path1, path2, atol=1e-6)
    assert np.allclose(path1, path3, atol=1e-6)


if __name__ == "__main__":
    test()
    print("test_floyd_warshall passed")
