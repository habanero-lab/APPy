# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    A = np.fromfunction(lambda i, j: ((i * j) % M) / M, (M, M), dtype=datatype)
    for i in range(M):
        A[i, i] = 1.0
    B = np.fromfunction(lambda i, j: ((N + i - j) % N) / N, (M, N),
                        dtype=datatype)

    return alpha, A, B


def kernel_np(alpha, A, B):
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha
    return B


import appy

@appy.jit(backend="triton", dump_code=True)
def kernel_appy(alpha, A, B):
    #pragma parallel for
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            s = 0.0
            #pragma simd
            for k in range(i+1, A.shape[0]):
                s += A[k, i] * B[k, j]
            B[i, j] += s
    B *= alpha
    return B


def test():
    N = 100

    alpha, A, B = initialize(N, N)
    B_ref = kernel_np(alpha, A, B)
    B = kernel_appy(alpha, A, B)
    assert np.allclose(B, B_ref, atol=1e-6)