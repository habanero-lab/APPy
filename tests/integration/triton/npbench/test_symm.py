# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i + j) % 100) / M, (M, N),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: ((N + i - j) % 100) / M, (M, N),
                        dtype=datatype)
    A = np.empty((M, M), dtype=datatype)
    for i in range(M):
        A[i, :i + 1] = np.fromfunction(lambda j: ((i + j) % 100) / M,
                                       (i + 1, ),
                                       dtype=datatype)
        A[i, i + 1:] = -999

    return alpha, beta, C, A, B


def kernel_np(alpha, beta, C, A, B):

    temp2 = np.empty((C.shape[1], ), dtype=C.dtype)
    C *= beta
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2
    return C


import appy


@appy.jit(backend="triton", dump_code=True)
def kernel_appy(alpha, beta, C, A, B):

    temp2 = np.empty((C.shape[1], ), dtype=C.dtype)
    C *= beta
    for i in range(C.shape[0]):
        #pragma parallel for
        for j in range(C.shape[1]):
            s = 0.0
            #pragma simd
            for k in range(i):
                C[k, j] += alpha * B[i, j] * A[i, k]
                s += B[k, j] * A[i, k]
            temp2[j] = s
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2
    return C


def test():
    M, N = 50, 50
    alpha, beta, C, A, B = initialize(M, N)
    C1, C2 = C.copy(), C.copy()

    kernel_np(alpha, beta, C1, A, B)
    kernel_appy(alpha, beta, C2, A, B)

    assert np.allclose(C1, C2)