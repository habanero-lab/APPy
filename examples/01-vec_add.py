import torch
import appy
from benchmark_utils import allclose, bench

@appy.jit(auto_transfer=False, dump_code=True) 
def kernel_appy(a, b):
    c = torch.empty_like(a)
    #pragma parallel for simd
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c


def kernel_lib(a, b):
    return a + b


def test():
    #torch.set_default_device('cuda')
    for N in [10000, 100000, 1000000, 10000000]:
        a = torch.randn(N)
        b = torch.randn(N)
        c_ref = kernel_lib(a, b)
        print(f"N: {a.shape[0]}, dtype: {a.dtype}")
        for f in [kernel_lib, kernel_appy]:
            c = f(a, b)
            assert allclose(c, c_ref, atol=1e-6)
            ms = bench(lambda: f(a, b))
            print(f"{f.__name__}: {ms:.4f} ms")


if __name__ == "__main__":
    test()
