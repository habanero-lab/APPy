# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i: (i + 2) / N, (N, ), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, (N, ), dtype=datatype)

    return A, B


def kernel_np(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
    return A, B

import appy

@appy.jit(backend="triton", dump_code=True)
def kernel_appy(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        #pragma parallel for simd
        for i in range(1, A.shape[0]-1):
            B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        
        #pragma parallel for simd
        for i in range(1, B.shape[0]-1):
            A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
    return A, B


def test():
    N = 10000 * 4
    A, B = initialize(N)

    A1, B1 = A.copy(), B.copy()
    kernel_np(1000, A1, B1)

    A2, B2 = A.copy(), B.copy()
    kernel_appy(1000, A2, B2)

    assert np.allclose(A1, A2, atol=1e-6)
    assert np.allclose(B1, B2, atol=1e-6)