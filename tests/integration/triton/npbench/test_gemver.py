# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, (N, ), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N, ), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N, ), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N, ), dtype=datatype)
    w = np.zeros((N, ), dtype=datatype)
    x = np.zeros((N, ), dtype=datatype)
    y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N, ), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N, ), dtype=datatype)

    return alpha, beta, A, u1, v1, u2, v2, w, x, y, z



def kernel_np(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):

    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
    return A, x, w


import appy
@appy.jit(backend="triton", dump_code=True)
def kernel_appy(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    M, N = A.shape
    #pragma parallel for
    for i in range(M):
        #pragma simd
        for j in range(N):
            A[i, j] += u1[i] * v1[j] + u2[i] * v2[j]

    #pragma parallel for
    for j in range(N):
        s = 0.0
        #pragma simd
        for i in range(M):
            s += beta * y[i] * A[i, j]
        x[j] += s + z[j]

    #pragma parallel for
    for i in range(M):
        s = 0.0
        #pragma simd
        for j in range(N):
            s += alpha * A[i, j] * x[j]
        w[i] += s
    return A, x, w


def test():
    alpha, beta, A, u1, v1, u2, v2, w, x, y, z = initialize(100)
    A1, x1, w1 = A.copy(), x.copy(), w.copy()
    A2, x2, w2 = A.copy(), x.copy(), w.copy()

    kernel_np(alpha, beta, A1, u1, v1, u2, v2, w1, x1, y, z)
    kernel_appy(alpha, beta, A2, u1, v1, u2, v2, w2, x2, y, z)

    assert np.allclose(A1, A2)
    assert np.allclose(x1, x2)
    assert np.allclose(w1, w2)