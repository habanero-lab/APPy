# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: i * (j + 2) / N, (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N, (N, N), dtype=datatype)

    return A, B


def kernel_np(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])
    return A, B


import appy

@appy.jit
def kernel_appy(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        #pragma parallel for
        for i in range(1, A.shape[0]-1):
            #pragma simd
            for j in range(1, A.shape[1]-1):
                B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] +
                                 A[i-1, j] + A[i+1, j])
        
        #pragma parallel for
        for i in range(1, B.shape[0]-1):
            #pragma simd
            for j in range(1, B.shape[1]-1):
                A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] +
                                 B[i-1, j] + B[i+1, j])
    return A, B


def test():
    N = 100
    A, B = initialize(N)
    A1, B1 = A.copy(), B.copy()
    kernel_np(10, A1, B1)

    A2, B2 = A.copy(), B.copy()
    kernel_appy(10, A2, B2)

    assert np.allclose(A1, A2, atol=1e-6)
    assert np.allclose(B1, B2, atol=1e-6)