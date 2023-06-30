import torch
import triton
import triton.language as tl
from slap import jit, max
import numba
from numba import prange
import numpy as np
from torch import arange, zeros, empty, sum, maximum, add, exp, t, mm

import torch.utils.benchmark as torchbench

def bench(fn):
    t0 = torchbench.Timer(
        stmt='fn()',
        globals={'fn': fn},
        num_threads=torch.get_num_threads()
    )
    return t0.timeit(20).mean * 1000

torch.set_default_device('cuda')

def mykernel(a, inner):
    n_vars, n_obs = a.shape
    u = torch.zeros([n_vars], dtype=a.dtype)
    b = torch.empty([n_vars, n_vars], dtype=a.dtype)
    inner(a, b, u, n_vars, n_obs)
    return b

@jit
def _mykernel(a, b, u, M, N, BM=8, BN=512):
    for i in range(M):  #pragma parallel
        s = 0.0
        for j in range(0, N, BN):
            #s += sum(a[i,j:j+BN]) / N  # somehow this triggers a segfault
            s += sum(a[i,j:j+BN] / N)
        b[i] = s

    for i in range(M):  #pragma parallel
        for j in range(M):  #pragma parallel
            cov = 0.0
            for k in range(0, N, BN):
                cov += sum((a[i,k:k+BN] - u[i]) * (a[j,k:k+BN] - u[j]))
            b[i,j] = cov / (N-1)

@jit
def _mykernel_blocked(a, b, u, M, N, BM=64, BN=64):
    for i in range(M):  #pragma parallel
        s = 0.0
        for j in range(0, N, 256):
            #s += sum(a[i,j:j+BN]) / N  # somehow this triggers a segfault
            s += sum(a[i,j:j+256] / N)
        b[i] = s

    for i in range(0, M, BM):  #pragma parallel
        for j in range(0, M, BM):  #pragma parallel
            cov = zeros([BM, BM], dtype=torch.float32)
            for k in range(0, N, BN):
                x = a[i:i+BM, k:k+BN] - u[i:i+BM][:,None]
                y = a[j:j+BM, k:k+BN] - u[j:j+BM][:,None]
                cov += mm(x, t(y))
            b[i:i+BM, j:j+BM] = cov / (N-1)


def numba_kernel(a):
    n_vars, n_obs = a.shape
    u = torch.zeros([n_vars], dtype=a.dtype)
    b = torch.empty([n_vars, n_vars], dtype=a.dtype)
    _numba_kernel(a.cpu().numpy(), b.cpu().numpy(), u.cpu().numpy(), n_vars, n_obs)
    return b

@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_kernel(a, b, u, M, N, BM=8, BN=512):
    for i in prange(M):  #pragma parallel
        s = 0.0
        for j in range(0, N, BN):
            #s += sum(a[i,j:j+BN]) / N  # somehow this triggers a segfault
            s += np.sum(a[i,j:j+BN] / N)
        b[i] = s

    for i in prange(M):  #pragma parallel
        for j in range(M):  #pragma parallel
            cov = 0.0
            for k in range(0, N, BN):
                cov += np.sum((a[i,k:k+BN] - u[i]) * (a[j,k:k+BN] - u[j]))
            b[i,j] = cov / (N-1)

def torch_kernel(a):
    #return torch.mean(a, dim=1)
    return torch.cov(a)

def test1():
    for dtype in [torch.float16]:
    #for dtype in [torch.float64]:
        for M, N in [(1024, 1024), (1024*2, 1024*2)]:
        #for M, N in [(1024, 256*2), (4096, 4096), (4096*4, 4096*4), (4096, 4096*8), (4096, 4096*16), (128, 4096*16), (256, 4096*16)]:
        #for M, N in [(8, 256)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, dtype=dtype)
            b_ref = torch_kernel(a)

            for f in (torch_kernel, _mykernel, _mykernel_blocked):
                ff = lambda: f(a)
                if f.__name__.startswith('_'):
                    ff = lambda: mykernel(a, f)
                b = ff()
                if not torch.allclose(b, b_ref, atol=0.1, rtol=0.1):
                    print(b)
                    print(b_ref)
                assert(torch.allclose(b, b_ref, atol=0.1, rtol=0.1))
                #ms, _, _ = triton.testing.do_bench(ff)
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()