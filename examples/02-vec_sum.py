import torch
import appy
from appy.utils import allclose, bench

# APPy tensors are `torch` tensors by default. Uncomment the 
# following lines to make APPy work with (and return) `cupy` tensors.
#import cupy
#appy.tensorlib = cupy

@appy.jit  # Comment this line to run the function in the Python interpter (debug mode)
def kernel_appy(a):
    ## Zero-initialize the output array
    b = torch.zeros(1, dtype=a.dtype)
    #pragma parallel for simd
    for i in range(a.shape[0]): 
        #pragma atomic
        b[0] += a[i]
    return b


def kernel_lib(a):
    return a.sum()


def test():
    torch.set_default_device('cuda')
    for N in [10000, 100000, 1000000, 10000000]:
        a = torch.randn(N)
        c_ref = kernel_lib(a)
        print(f"N: {a.shape[0]}, dtype: {a.dtype}")
        for f in [kernel_lib, kernel_appy]:
            c = f(a)
            assert allclose(c, c_ref, atol=1e-6)
            ms = bench(lambda: f(a))
            print(f"{f.__name__}: {ms:.4f} ms")

if __name__ == '__main__':
    test()