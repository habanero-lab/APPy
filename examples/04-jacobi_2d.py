import torch
import appy
import copy
from appy.utils import allclose, bench

torch.set_default_device('cuda')

@appy.jit(auto_simd=True)
def kernel_appy(A, B):
    M, N = A.shape
    for t in range(1, 30):
        #pragma 1:M-1=>parallel 1:N-1=>parallel
        B[1:M-1, 1:N-1] = 0.2 * (A[1:M-1, 1:N-1] + A[1:M-1, :N-2] + A[1:M-1, 2:N] +
                                A[2:M, 1:N-1] + A[0:M-2, 1:N-1])
        #pragma 1:M-1=>parallel 1:N-1=>parallel
        A[1:M-1, 1:N-1] = 0.2 * (B[1:M-1, 1:N-1] + B[1:M-1, :N-2] + B[1:M-1, 2:N] +
                                B[2:M, 1:N-1] + B[0:M-2, 1:N-1])
    return A, B


def kernel_lib(A, B):
    for t in range(1, 30):        
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])
    return A, B


def test():
    for N in [1000, 4000]:
        A = torch.randn(N, N)
        B = torch.randn(N, N)
        A_ref, B_ref = kernel_lib(copy.deepcopy(A), copy.deepcopy(B))
        print(f"shape: {A.shape}, dtype: {A.dtype}")
        for f in [kernel_lib, kernel_appy]:
            A_, B_ = f(copy.deepcopy(A), copy.deepcopy(B))
            assert allclose(A_, A_ref, atol=1e-6)
            #A_copy, B_copy = copy.deepcopy(A), copy.deepcopy(B)
            #ms = appy.utils.bench(lambda: f(A_copy, B_copy))
            ms = bench(lambda: f(copy.deepcopy(A), copy.deepcopy(B)))
            print(f"{f.__name__}: {ms:.4f} ms")


if __name__ == "__main__":
    test()
