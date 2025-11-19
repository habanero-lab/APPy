# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, N),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N, (N, N),
                        dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)

    return alpha, beta, A, B, x



def kernel_np(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x


import appy

@appy.jit(backend="triton", dump_code=True)
def kernel_appy(alpha, beta, A, B, x):    
    y = np.empty((A.shape[0], ), dtype=x.dtype)
    #pragma parallel for
    for i in range(A.shape[0]):
        s = 0.0
        #pragma simd
        for j in range(A.shape[1]):
            s += alpha * A[i, j] * x[j] + beta * B[i, j] * x[j]
        y[i] = s
    return y


def test_float64():
    M = 100
    N = 100
    alpha, beta, A, B, x = initialize(N, datatype=np.float64)
    y_ref = kernel_np(alpha, beta, A, B, x)
    y_appy = kernel_appy(alpha, beta, A, B, x)
    assert np.allclose(y_ref, y_appy)