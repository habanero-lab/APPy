import torch
from appy.utils import bench
from appy import jit, max
from torch import arange, zeros, empty, sum, maximum, add, exp

torch.set_default_device('cuda')

def mykernel(a, M, N, inner):
    t0 = empty([M], dtype=a.dtype)
    t1 = empty([M, N], dtype=a.dtype)
    t2 = empty([M], dtype=a.dtype)
    b = empty([M, N], dtype=a.dtype)
    inner(a, b, t0, t1, t2, M, N)
    return b

@jit
def _mykernel(a, b, t0, t1, t2, M, N):
    #pragma parallel
    for i in range(M):  
        _a = a[i,0:N]
        t0 = max(_a)
        t1 = exp(_a - t0)
        t2 = sum(t1)
        _b = t1 / t2
        b[i,0:N] = _b

#@jit
def _mykernel1(a, b, t0, t1, t2, M, N):
    #pragma :M=>p :N=>b(N)
    t0[:M] = max(a[:M, :N], axis=1)
    t1[:M, :N] = exp(a[:M, :N] - t0[:M, None])
    t2[:M] = sum(t1[:M, :N], axis=1)
    b[:M, :N] = t1[:M, :N] / t2[:M, None]

def torch_kernel_native(a, M, N):
    return torch.softmax(a, dim=1)

#@torch.compile
def torch_kernel(a, M, N):
    t0 = max(a, axis=1)
    t1 = exp(a - t0[:,None])
    t2 = sum(t1, axis=1)
    b = t1 / t2[:,None]
    return b

def test1():
    for dtype in [torch.float32]:  # torch.float16 has error: 'llvm.fmul' op requires the same type for all operands and results
        for M, N in [(4096, 4096), (4096*4, 4096*4), (4096, 4096*8)]:
        #for M, N in [(4096, 4096*8)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, dtype=dtype)
            b_ref = torch_kernel(a, M, N)

            for f in (torch_kernel, torch_kernel_native, _mykernel, _mykernel1):
                ff = lambda: f(a, M, N)
                if f.__name__.startswith('_'):
                    ff = lambda: mykernel(a, M, N, f)
                b = ff()
                assert(torch.allclose(b, b_ref, atol=0.5, rtol=0.05))
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()