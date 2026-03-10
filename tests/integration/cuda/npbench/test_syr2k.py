# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import appy


def initialize(M, N, datatype=np.float32):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 3) % N) / M, (N, N), dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, M), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % M) / M, (N, M), dtype=datatype)
    return alpha, beta, C, A, B


def kernel_np(alpha, beta, C, A, B):
    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] +
                             B[:i + 1, k] * alpha * A[i, k])
    return C


@appy.jit(backend="cuda", dump_code=False)
def kernel_appy(alpha, beta, C, A, B):
    #pragma parallel for
    for i in range(A.shape[0]):
        #pragma simd
        for j in range(i + 1):
            C[i, j] *= beta
        for k in range(A.shape[1]):
            #pragma simd
            for j in range(i + 1):
                C[i, j] += (A[j, k] * alpha * B[i, k] +
                            B[j, k] * alpha * A[i, k])
    return C


@appy.jit(backend="cuda", dump_code=False)
def kernel_appy1(alpha, beta, C, A, B):
    #pragma parallel for
    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] +
                             B[:i + 1, k] * alpha * A[i, k])
    return C


def test():
    M = 100
    N = 100
    alpha, beta, C, A, B = initialize(M, N)
    A = A.copy()
    B = B.copy()
    C1, C2, C3 = np.copy(C), C.copy(), C.copy()

    kernel_np(alpha, beta, C1, A, B)
    kernel_appy(alpha, beta, C2, A, B)
    kernel_appy1(alpha, beta, C3, A, B)

    assert np.allclose(C1, C2)
    assert np.allclose(C1, C3)


if __name__ == "__main__":
    test()
    print("test_syr2k passed")
