# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import appy.np_shared as nps


def initialize(M, N, datatype=np.float32):
    A = np.fromfunction(lambda i, j: (((i - j) % M) / M) * 3, (M, N), dtype=datatype)
    return A


def kernel_np(A):
    M, N = A.shape
    Q = np.zeros((M, N), dtype=A.dtype)
    R = np.zeros((N, N), dtype=A.dtype)
    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]
    return Q, R


import appy


@appy.jit(dump_code=False)
def kernel_appy(A):
    M, N = A.shape
    Q = np.zeros((M, N), dtype=A.dtype)
    R = np.zeros((N, N), dtype=A.dtype)
    for k in range(N):
        nrm = np.sum(A[:, k] * A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        #pragma parallel for
        for j in range(k + 1, N):
            s = np.sum(Q[:, k] * A[:, j])
            R[k, j] = s
            A[:, j] -= Q[:, k] * s
    return Q, R


def test():
    M, N = 60, 40

    A = initialize(M, N)
    A1 = A.copy()
    Q1, R1 = kernel_np(A1)

    A2 = nps.copy(A)
    Q2, R2 = kernel_appy(A2)

    assert np.allclose(Q1, Q2, atol=1e-4), f"Q: max err {np.max(np.abs(Q1 - Q2))}"
    assert np.allclose(R1, R2, atol=1e-4), f"R: max err {np.max(np.abs(R1 - R2))}"
