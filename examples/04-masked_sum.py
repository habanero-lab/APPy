import torch
import appy
from benchmark_utils import allclose, bench

torch.set_default_device('cuda')

@appy.jit  
def kernel_appy1(a, mask):
    b = 0.0
    #pragma parallel for shared(b)
    for i in range(a.shape[0]):
        if mask[i]:  ## simd directive is not support for loops with control flows
            b += a[i]
    return b


@appy.jit  
def kernel_appy2(a, mask):
    b = 0.0
    #pragma parallel for simd shared(b)
    for i in range(a.shape[0]):
        b += a[i] if mask[i] else 0.0  # Use ternary operator instead
    return b


def kernel_lib(a, mask):
    return a[mask].sum()


def test():
    for N in [10000, 100000, 1000000, 10000000]:
        a = torch.randn(N)
        mask = torch.randint(0, 2, size=(N,)) > 0
        c_ref = kernel_lib(a, mask)
        print(f"N: {a.shape[0]}, dtype: {a.dtype}")
        for f in [kernel_lib, kernel_appy1, kernel_appy2]:
            c = f(a, mask)
            assert allclose(c, c_ref, atol=1e-6)
            ms = bench(lambda: f(a, mask))
            print(f"{f.__name__}: {ms:.4f} ms")

if __name__ == '__main__':
    test()