# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 2) % N) / M, (N, N),
                        dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, M),
                        dtype=datatype)

    return alpha, beta, C, A


def kernel_np(alpha, beta, C, A):

    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]
    return C


import appy

@appy.jit(backend="triton", dump_code=True)
def kernel_appy(alpha, beta, C, A):
    #pragma parallel for
    for i in range(A.shape[0]):
        #pragma simd
        for j in range(i + 1):
            C[i, j] *= beta
        for k in range(A.shape[1]): 
            #pragma simd
            for j in range(i + 1):
                C[i, j] += alpha * A[i, k] * A[j, k]
    return C


def test():
    M = 100
    N = 100
    alpha, beta, C, A = initialize(M, N)
    C1, C2 = C.copy(), C.copy()

    kernel_np(alpha, beta, C1, A)
    kernel_appy(alpha, beta, C2, A)

    assert np.allclose(C1, C2)