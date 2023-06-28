import torch
import triton
import triton.language as tl
from slap import jit, max
from torch import arange, zeros, empty, sum, maximum, add, exp, where

torch.set_default_device('cuda')

def mykernel(a, M, N, inner):
    b = empty([M, N], dtype=a.dtype)
    inner(a, b, M, N)
    return b

@jit
def _mykernel(a, b, M, N, BN=256):
    for i in range(M):  #pragma parallel
        for j in range(N):  #pragma parallel block(BN)
            a_ij = a[i,j]
            b[i,j] = where(a[i,j] > 0, a[i,j], 0.01*a[i,j])

def torch_kernel(a, M, N):
    return where(a > 0, a, 0.01 * a)

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        #for M, N in [(1024, 1024), (4096, 4096), (4096*4, 4096*4), (4096, 4096*8), (4096, 4096*16), (128, 4096*16), (256, 4096*16)]:
        for M, N in [(128, 128), (4096, 4096*8), (128, 4096*8)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, dtype=dtype)
            b_ref = torch_kernel(a, M, N)

            for f in (torch_kernel, _mykernel):
                ff = lambda: f(a, M, N)
                if f.__name__.startswith('_'):
                    ff = lambda: mykernel(a, M, N, f)
                b = ff()
                assert(torch.allclose(b, b_ref, atol=0.1, rtol=0.1))
                ms, _, _ = triton.testing.do_bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()