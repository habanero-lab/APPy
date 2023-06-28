import torch
import triton
import triton.language as tl
from slap import jit, max
from torch import arange, zeros, empty, sum, maximum, add, exp

torch.set_default_device('cuda')

def mykernel_op(a, M, N):
    t0 = max(a, axis=1)
    t1 = exp(a - t0[:,None])
    t2 = sum(t1, axis=1)
    b = t1 / t2[:,None]
    return b

def mykernel(a, M, N, inner):
    t0 = empty([M], dtype=a.dtype)
    t0.fill_(float('-inf'))
    t1 = empty([M, N], dtype=a.dtype)
    t2 = empty([M], dtype=a.dtype)
    t2.fill_(0)
    b = empty([M, N], dtype=a.dtype)
    inner(a, b, t0, t1, t2, M, N)
    return b

def _mykernel_unfused(a, b, t0, t1, t2, M, N, BLOCK=128):
    for i in range(M):  #pragma parallel
        for j in range(N):  #pragma parallel block(BLOCK) reduction(max:t0)
            t0[i] = max(t0[i], a[i,j])

    for i in range(M):  #pragma parallel
        for j in range(N):  #pragma parallel block(BLOCK)
            t1[i,j] = exp(a[i,j] - t0[i])

    for i in range(M):  #pragma parallel
        for j in range(N):  #pragma parallel block(BLOCK) reduction(+:t2)
            t2[i] += t1[i,j]

    for i in range(M):  #pragma parallel
        for j in range(N):  #pragma parallel block(BLOCK)
            b[i,j] = t1[i,j] / t2[i]


def _mykernel_max_parallelism(a, b, t0, t1, t2, M, N, BN=128):
    for i in range(M):  #pragma parallel
        for j in range(0, N, BN):  #pragma parallel reduction(t0)
            t0[i] = maximum(t0[i], max(a[i,j:j+BN]))

    for i in range(M):  #pragma parallel
        for j in range(0, N, BN):  #pragma parallel reduction(t2)
            _t1 = exp(a[i,j:j+BN] - t0[i])
            t1[i,j:j+BN] = _t1
            t2[i] += sum(_t1)

    for i in range(M):  #pragma parallel
        for j in range(0, N, BN):  #pragma parallel 
            b[i,j:j+BN] = t1[i,j:j+BN] / t2[i]


@jit
def _mykernel_max_locality(a, b, t0, t1, t2, M, N, BN=256):
    #pragma parallel
    for i in range(M):  
        for j in range(0, N, BN):
            t0[i] = maximum(t0[i], max(a[i,j:j+BN]))

        for j in range(0, N, BN): 
            _t1 = exp(a[i,j:j+BN] - t0[i])
            t1[i,j:j+BN] = _t1
            t2[i] += sum(_t1)

        for j in range(0, N, BN):
            b[i,j:j+BN] = t1[i,j:j+BN] / t2[i]

@jit
def _mykernel_max_locality_full_block_col(a, b, t0, t1, t2, M, N):
    for i in range(M):  #pragma parallel
        t0[0] = max(a[i,0:N])
        t1[0,0:N] = exp(a[i,0:N] - t0[0])
        t2[0] = sum(t1[0,0:N])
        b[i,0:N] = t1[0,0:N] / t2[0]

@jit
def _mykernel_max_locality_cache_row(a, b, t0, t1, t2, M, N):
    for i in range(M):  #pragma parallel
        _a = a[i,0:N]
        _t0 = max(_a)
        _t1 = exp(_a - _t0)
        _t2 = sum(_t1)
        _b = _t1 / _t2
        b[i,0:N] = _b

def torch_kernel(a, M, N):
    return torch.softmax(a, dim=1)

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        for M, N in [(4096, 4096), (4096*4, 4096*4), (4096, 4096*8), (4096, 4096*16), (128, 4096*16)]:
        #for M, N in [(4096, 4096*8)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, dtype=dtype)
            b_ref = torch_kernel(a, M, N)

            for f in (torch_kernel, _mykernel_max_locality_cache_row, _mykernel_max_locality, _mykernel_max_parallelism):
                ff = lambda: f(a, M, N)
                if f.__name__.startswith('_'):
                    ff = lambda: mykernel(a, M, N, f)
                b = ff()
                assert(torch.allclose(b, b_ref, atol=0.5, rtol=0.1))
                ms, _, _ = triton.testing.do_bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()