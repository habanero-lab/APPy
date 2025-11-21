# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)

    A = rng.random((M, N), dtype=datatype)
    while np.linalg.matrix_rank(A) < N:
        A = rng.random((M, N), dtype=datatype)

    return A


def kernel_np(A):

    Q = np.zeros_like(A)
    R = np.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    for k in range(A.shape[1]):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R


import appy

@appy.jit(backend="triton", dump_code=True)
def kernel(A):

    Q = np.zeros_like(A)
    R = np.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    for k in range(A.shape[1]):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        #pragma parallel for
        for j in range(k + 1, A.shape[1]):
            s = 0.0
            #pragma simd
            for i in range(A.shape[0]):
                s += Q[i, k] * A[i, j]
            
            #pragma simd
            for i in range(A.shape[0]):
                A[i, j] -= Q[i, k] * s
            R[k, j] = s

    return Q, R


def test():
    M = 100
    N = 100
    A = initialize(M, N)

    A1, A2 = A.copy(), A.copy()

    Q1, R1 = kernel_np(A1)
    Q2, R2 = kernel(A2)

    assert np.allclose(Q1, Q2)
    assert np.allclose(R1, R2)