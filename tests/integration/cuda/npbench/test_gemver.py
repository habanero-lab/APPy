# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import appy


def initialize(N, datatype=np.float32):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, (N,), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N,), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N,), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N,), dtype=datatype)
    w = np.zeros((N,), dtype=datatype)
    x = np.zeros((N,), dtype=datatype)
    yv = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N,), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N,), dtype=datatype)
    return alpha, beta, A, u1, u2, v1, v2, w, x, yv, z


def kernel_np(alpha, beta, A, u1, u2, v1, v2, w, x, yv, z):
    A = A + np.outer(u1, v1) + np.outer(u2, v2)
    x = x + beta * yv @ A + z
    w = w + alpha * A @ x
    return A, x, w


@appy.jit(backend="cuda", dump_code=False)
def kernel_appy(alpha, beta, A, u1, u2, v1, v2, w, x, yv, z):
    M, N = A.shape
    #pragma parallel for
    for i in range(M):
        A[i, :] += u1[i] * v1[:] + u2[i] * v2[:]

    #pragma parallel for
    for j in range(N):
        s = np.sum(beta * yv[:] * A[:, j])
        x[j] = x[j] + s + z[j]

    #pragma parallel for
    for i in range(M):
        s = np.sum(alpha * A[i, :] * x[:])
        w[i] = w[i] + s

    return A, x, w


def test():
    N = 100
    alpha, beta, A, u1, u2, v1, v2, w, x, yv, z = initialize(N)

    A1, w1, x1 = A.copy(), w.copy(), x.copy()
    A1_ref, x1_ref, w1_ref = kernel_np(alpha, beta, A1, u1, u2, v1, v2, w1, x1, yv, z)

    A2 = A.copy()
    u1_c, u2_c = u1.copy(), u2.copy()
    v1_c, v2_c = v1.copy(), v2.copy()
    w2, x2 = w.copy(), x.copy()
    yv_c, z_c = yv.copy(), z.copy()
    A2_out, x2_out, w2_out = kernel_appy(alpha, beta, A2, u1_c, u2_c, v1_c, v2_c, w2, x2, yv_c, z_c)

    assert np.allclose(A1_ref, A2_out, atol=1e-4), f"A: max err {np.max(np.abs(A1_ref - A2_out))}"
    assert np.allclose(x1_ref, x2_out, atol=1e-4), f"x: max err {np.max(np.abs(x1_ref - x2_out))}"
    assert np.allclose(w1_ref, w2_out, atol=1e-4), f"w: max err {np.max(np.abs(w1_ref - w2_out))}"


if __name__ == "__main__":
    test()
    print("test_gemver passed")
