import torch
import appy
from appy.utils import allclose, bench 

torch.set_default_device('cuda')
# import cupy
# appy.tensorlib = cupy

@appy.jit(auto_simd=True)
def kernel_appy(alpha, A, x):
    M, N = A.shape
    y = torch.empty(M, dtype=A.dtype)
    ## parallel reduction in tensor-oriented model is not supported yet
    #pragma :M=>parallel :N=>reduce(sum)
    y[:M] = appy.mv(alpha * A[:M, :N], x[:N])
    return y


def kernel_lib(alpha, A, x):
    y = alpha * A @ x
    return y
    

def test():
    for N in [1000, 4000, 8000]:
        A = torch.randn(N, N)
        x = torch.randn(N)
        alpha = 1.0
        y_ref = kernel_lib(alpha, A, x)
        for f in [kernel_lib, kernel_appy]:
            y = f(alpha, A, x)
            assert allclose(y, y_ref, atol=1e-6)
            ms = bench(lambda: f(alpha, A, x))
            print(f"{f.__name__}: {ms:.4f} ms")


if __name__ == "__main__":
    test()
