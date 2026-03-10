"""
Tests that torch CUDA tensors are passed zero-copy to CUDA kernels.
No host<->device transfers should occur for inputs already on GPU.
"""

import numpy as np
import torch
import appy


@appy.jit(backend="cuda", dump_code=False)
def vec_add(a, b, c):
    #pragma parallel for
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]


@appy.jit(backend="cuda", dump_code=False)
def saxpy(a, x, y):
    #pragma parallel for
    for i in range(x.shape[0]):
        y[i] = a * x[i] + y[i]


@appy.jit(backend="cuda", dump_code=False)
def mat_add(A, B, C):
    #pragma parallel for
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] + B[i, j]


def test_vec_add_torch():
    N = 1024
    a = torch.rand(N, dtype=torch.float32, device='cuda')
    b = torch.rand(N, dtype=torch.float32, device='cuda')
    c = torch.zeros(N, dtype=torch.float32, device='cuda')

    # Record pointers before — they must not change (zero-copy)
    ptr_a = a.data_ptr()
    ptr_b = b.data_ptr()
    ptr_c = c.data_ptr()

    vec_add(a, b, c)

    assert a.data_ptr() == ptr_a
    assert b.data_ptr() == ptr_b
    assert c.data_ptr() == ptr_c

    ref = a + b
    assert torch.allclose(c, ref, atol=1e-6), f"Max error: {(c - ref).abs().max()}"
    print("test_vec_add_torch passed")


def test_saxpy_torch():
    N = 1_000_000
    a = np.float32(2.5)
    x = torch.rand(N, dtype=torch.float32, device='cuda')
    y = torch.rand(N, dtype=torch.float32, device='cuda')
    ref = a * x + y

    ptr_x = x.data_ptr()
    ptr_y = y.data_ptr()

    saxpy(a, x, y)

    assert x.data_ptr() == ptr_x
    assert y.data_ptr() == ptr_y
    assert torch.allclose(y, ref, atol=1e-5), f"Max error: {(y - ref).abs().max()}"
    print("test_saxpy_torch passed")


def test_mat_add_torch():
    M, N = 512, 512
    A = torch.rand(M, N, dtype=torch.float32, device='cuda')
    B = torch.rand(M, N, dtype=torch.float32, device='cuda')
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    ptr_A = A.data_ptr()
    ptr_B = B.data_ptr()
    ptr_C = C.data_ptr()

    mat_add(A, B, C)

    assert A.data_ptr() == ptr_A
    assert B.data_ptr() == ptr_B
    assert C.data_ptr() == ptr_C

    ref = A + B
    assert torch.allclose(C, ref, atol=1e-6), f"Max error: {(C - ref).abs().max()}"
    print("test_mat_add_torch passed")


def test_mixed_numpy_torch():
    """Mix of numpy (auto-migrated) and torch CUDA (zero-copy) inputs."""
    N = 1024
    a = np.random.rand(N).astype(np.float32)       # numpy — will be migrated
    b = torch.rand(N, dtype=torch.float32, device='cuda')  # torch — zero-copy
    c = torch.zeros(N, dtype=torch.float32, device='cuda')

    ptr_b = b.data_ptr()
    ptr_c = c.data_ptr()

    vec_add(a, b, c)

    assert b.data_ptr() == ptr_b
    assert c.data_ptr() == ptr_c

    ref = torch.from_numpy(a).cuda() + b
    assert torch.allclose(c, ref, atol=1e-6), f"Max error: {(c - ref).abs().max()}"
    print("test_mixed_numpy_torch passed")


if __name__ == "__main__":
    test_vec_add_torch()
    test_saxpy_torch()
    test_mat_add_torch()
    test_mixed_numpy_torch()
