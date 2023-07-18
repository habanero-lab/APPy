import torch
import triton
import triton.language as tl
from appy import jit, max
from torch import arange, zeros, empty, sum, maximum, add, exp

torch.set_default_device('cuda')

def mykernel(a, inner):
    n_vars, n_obs = a.shape
    b = torch.zeros([n_vars], dtype=a.dtype)
    inner(a, b, n_vars, n_obs)
    return b

@jit
def _mykernel(a, b, M, N, BN=256):
    for i in range(M):  #pragma parallel
        for j in range(0, N, BN):
            b[i] += sum(a[i,j:j+BN] / N)

@jit
def _mykernel1(a, b, M, N, BN=256):
    #pragma par_dim(:M) seq_dim(:N:BN)
    b[:M] = sum(a[:M, :N]) / N

def torch_kernel(a):
    return torch.mean(a, dim=1)
    #return torch.cov(a)

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        for M, N in [(1024, 1024), (4096, 4096), (4096*4, 4096*4), (4096, 4096*8), (4096, 4096*16), (128, 4096*16), (256, 4096*16)]:
        #for M, N in [(8, 256)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, dtype=dtype)
            b_ref = torch_kernel(a)

            for f in (torch_kernel, _mykernel):
                ff = lambda: f(a)
                if f.__name__.startswith('_'):
                    ff = lambda: mykernel(a, f)
                b = ff()
                #print(b)
                #print(b_ref)
                assert(torch.allclose(b, b_ref, atol=0.1, rtol=0.1))
                ms, _, _ = triton.testing.do_bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()