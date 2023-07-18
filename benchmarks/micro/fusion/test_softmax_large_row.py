import torch
from appy import jit, max
from appy.utils import bench
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
def _mykernel(a, b, t0, t1, t2, M, N, BN=512):
    #pragma parallel
    for i in range(M):
        m = 0-10000.0
        for j in range(0, N, BN):
            m = maximum(m, max(a[i,j:j+BN]))

        s = 0.0
        for j in range(0, N, BN): 
            s += sum(exp(a[i,j:j+BN] - m))

        for j in range(0, N, BN):
            b[i,j:j+BN] = (exp(a[i,j:j+BN] - m)) / s

#@jit
def _mykernel1(a, b, t0, t1, t2, M, N, BN=512):
    #pragma :M=>p :N=>block(BN),reduce(max:t0,+:t2)
    t0[:M] = max(a[:M, :N], axis=1)
    t2[:M] = sum(exp(a[:M, :N] - t0[:M, None]), axis=1)
    b[:M, :N] = (exp(a[:M, :N] - t0[:M, None])) / t2[:M, None]

def _mykernel2(a, b, t0, t1, t2, M, N, BN=512):
    #pragma parallel
    for i in range(M):
        m = max(a[i,:])
        s = sum(exp(a[i,:] - m))
        b[i,:] = (exp(a[i,:] - m)) / s

def torch_kernel_native(a, M, N):
    return torch.softmax(a, dim=1)

@torch.compile
def torch_kernel(a, M, N):
    t0 = max(a, axis=1)
    t1 = exp(a - t0[:,None])
    t2 = sum(t1, axis=1)
    b = t1 / t2[:,None]
    return b

def test1():
    for dtype in [torch.float32]:  # torch.float16 has error: 'llvm.fmul' op requires the same type for all operands and results
        for M, N in [(4096, 4096*16), (4096, 4096*32)]:
        #for M, N in [(4096, 4096*8)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, dtype=dtype)
            b_ref = torch_kernel(a, M, N)

            for f in (torch_kernel, torch_kernel_native, _mykernel):
                ff = lambda: f(a, M, N)
                if f.__name__.startswith('_'):
                    ff = lambda: mykernel(a, M, N, f)
                b = ff()
                assert(torch.allclose(b, b_ref, atol=0.5, rtol=0.05))
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()