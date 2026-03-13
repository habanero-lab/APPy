# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import appy.np_shared as nps


def initialize(M, N, datatype=np.float32):
    alpha = datatype(1.5)
    A = np.fromfunction(lambda i, j: ((i * j) % M) / M, (M, M), dtype=datatype)
    for i in range(M):
        A[i, i] = 1.0
    B = np.fromfunction(lambda i, j: ((N + i - j) % N) / N, (M, N), dtype=datatype)
    return alpha, A, B


def kernel_np(alpha, A, B):
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha
    return B


import appy


@appy.jit(dump_code=False)
def kernel_appy(alpha, A, B):
    M, N = B.shape
    for i in range(M):
        #pragma parallel for
        for j in range(N):
            B[i, j] += A[i + 1:, i] @ B[i + 1:, j]
    B *= alpha
    return B


def test():
    N = 100
    alpha, A, B = initialize(N, N)

    B1 = B.copy()
    kernel_np(alpha, A, B1)

    A_m, B_m = nps.copy(A), nps.copy(B)
    kernel_appy(alpha, A_m, B_m)

    assert np.allclose(B1, B_m, atol=1e-4), f"max err {np.max(np.abs(B1 - B_m))}"
