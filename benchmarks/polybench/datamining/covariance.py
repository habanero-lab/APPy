import torch
from slap import jit, max
import numpy as np
import numba
from torch import arange, zeros, empty, sum, maximum, add, exp, t, mm

from slap.utils import bench

torch.set_default_device('cuda')

def mykernel(a, inner):
    n_vars, n_obs = a.shape
    u = torch.empty([n_vars], dtype=a.dtype)
    b = torch.empty([n_vars, n_vars], dtype=a.dtype)
    inner(a, b, u, n_vars, n_obs)
    return b

@jit
def _mykernel(a, b, u, M, N, BN=512):
    #pragma parallel :N=>block(BN)
    for i in range(M):  
        mean = sum(a[i,:N])
        mean /= N
        a[i:N] -= mean

        for j in range(i):  
            b[i,j] = sum(a[i,:N] * a[j,:N])
            b[i,j] /= N - 1
            b[j,i] = b[i,j]
            
@jit
def _mykernel_blocked(a, b, u, M, N, BM=8, BN=512):
    #pragma parallel
    for i in range(M):  
        mean = sum(a[i,:N]) / N
        a[i:N] -= mean

    #pragma parallel
    for i in range(0, M, BM):
        #pragma parallel
        for j in range(0, M, BM):  
            b[i:i+BM,j:j+BM] = a[i:i+BM,:N] @ t(a[j:j+BM,:N])
            b[i:i+BM,j:j+BM] /= N - 1

def numba_kernel(a):
    n_vars, n_obs = a.shape
    u = torch.zeros([n_vars], dtype=a.dtype)
    b = torch.empty([n_vars, n_vars], dtype=a.dtype)
    _numba_kernel(a.cpu().numpy(), b.cpu().numpy(), u.cpu().numpy(), n_vars, n_obs)
    return b

@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_kernel(a, b, u, M, N):
    for i in numba.prange(M):  
        mean = np.sum(a[i,:N])
        mean /= N
        a[i:N] -= mean

        for j in range(i):  
            b[i,j] = np.sum(a[i,:N] * a[j,:N])
            b[i,j] /= N - 1
            b[j,i] = b[i,j]

def _torch_kernel(a, b, u, M, N, BN=512):
    for i in range(M):  
        mean = sum(a[i,:N])
        mean /= N
        a[i:N] -= mean

        for j in range(i):  
            b[i,j] = sum(a[i,:N] * a[j,:N])
            b[i,j] /= N - 1
            b[j,i] = b[i,j]

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        for M, N in [(1024, 1024), (1024*4, 1024*4), (1024*16, 1024*16)]:
        #for M, N in [(1024, 256*2), (4096, 4096), (4096*4, 4096*4), (4096, 4096*8), (4096, 4096*16), (128, 4096*16), (256, 4096*16)]:
        #for M, N in [(8, 256)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, dtype=dtype)
            b_ref = torch_kernel(a)

            for f in (torch_kernel, _mykernel_2_loops, _mykernel_3_loops,  _mykernel_blocked, _mykernel_ops):
            #for f in (torch_kernel, _mykernel_2_loops):
                ff = lambda: f(a)
                if f.__name__.startswith('_'):
                    ff = lambda: mykernel(a, f)
                b = ff()
                if not torch.allclose(b, b_ref, atol=0.1, rtol=0.01):
                    print(b)
                    print(b_ref)
                assert(torch.allclose(b, b_ref, atol=0.1, rtol=0.01))
                #ms, _, _ = triton.testing.do_bench(ff)
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()