# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import appy


def initialize(N, datatype=np.float32):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N, (N, N), dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, (N,), dtype=datatype)
    return alpha, beta, A, B, x


def kernel_np(alpha, beta, A, B, x):
    return alpha * A @ x + beta * B @ x


@appy.jit(backend="cuda", dump_code=False)
def kernel_appy(alpha, beta, A, B, x):
    y = np.empty((A.shape[0],), dtype=x.dtype)
    #pragma parallel for
    for i in range(A.shape[0]):
        y[i] = np.sum(alpha * A[i, :] * x[:] + beta * B[i, :] * x[:])
    return y


def test():
    N = 100
    alpha, beta, A, B, x = initialize(N)
    y_ref = kernel_np(alpha, beta, A, B, x).astype(np.float32)

    y = kernel_appy(alpha, beta, A.copy(), B.copy(), x.copy())
    assert np.allclose(y_ref, y, atol=1e-4), f"max err {np.max(np.abs(y_ref - y))}"


if __name__ == "__main__":
    test()
    print("test_gesummv passed")
