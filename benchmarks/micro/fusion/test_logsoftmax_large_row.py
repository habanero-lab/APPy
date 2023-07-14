import torch
from slap.utils import bench
from slap import jit, max
from torch import arange, zeros, empty, sum, maximum, add, exp, log

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
    t0.fill_(float('-inf'))
    t2.fill_(0)
    #pragma parallel
    for i in range(M):
        for j in range(0, N, BN):
            t0[i] = maximum(t0[i], max(a[i,j:j+BN]))

        for j in range(0, N, BN): 
            t2[i] += sum(exp(a[i,j:j+BN] - t0[i]))

        for j in range(0, N, BN):
            b[i,j:j+BN] = a[i,j:j+BN] - (t0[i] + log(t2[i]))

#@jit
def _mykernel1(a, b, t0, t1, t2, M, N):
    #pragma :M=>p :N=>block(BN),reduce(max:m,+:s)
    t0[:M] = max(a[:M, :N], axis=1)
    t2[:M] = sum(exp(a[:M, :N] - t0[:M, None]), axis=1)
    b[:M, :N] = a[:M, :N] - (t0[:M] + log(t2[:M]))[:M, None]

def _mykernel1_loop(a, b, t0, t1, t2, M, N, BN=512):
    #pragma parallel loop :N=>block(BN),reduce(max:m,+:s)
    for i in range(0, M, BM):
        m = max(a[i, :N])
        s = sum(exp(a[i, :N] - m))
        b[i, :N] = a[i, :N] - (m + log(s))

# Our version above is 5 lines of code
# Typical CUDA version is about 170 lines of code: https://github.com/pytorch/pytorch/blob/d49cb6613eac9ad4d9dde60243b6c981d3952094/LogSoftMax.cu#L134


def torch_kernel_native(a, M, N):
    return torch.log_softmax(a, dim=1)

@torch.compile
def torch_kernel(a, M, N):
    t0 = max(a, axis=1)
    t1 = exp(a - t0[:,None])
    t2 = log(sum(t1, axis=1))
    b = a - (t0 + t2)[:,None]
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