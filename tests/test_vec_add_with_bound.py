import numpy as np
import torch
import triton
from slap import parallel
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule

from torch import arange, zeros, empty

#@slap.jit
def kernel(a, b, c, N, BLOCK: parallel):
    for i in range(0, N, BLOCK):  #pragma parallel
        ii = arange(i, i+BLOCK)
        ii = ii[ii < N]
        c[ii] = a[ii] + b[ii]

def kernel_compiled(a, b, c, N, BLOCK: parallel):
    if not hasattr(kernel_compiled, 'cached'):
        kernel_compiled.cached = {}
    
    args = (a, b, c, BLOCK)
    if args not in kernel_compiled.cached:
        mod = SourceModule("""
            __global__ void _kernel(float *c, float *a, float *b, int N) {
                int i = blockIdx.x * blockDim.x;
                int ii = i + threadIdx.x;
                if (ii < N) {
                    c[ii] = a[ii] + b[ii];
                }
            }
        """, options=['-O3'])
        _kernel = mod.get_function("_kernel")
        kernel_compiled.cached[args] = _kernel

    _kernel = kernel_compiled.cached[args]
    nthreads = 128
    block = (nthreads, 1, 1)
    grid = ((N+nthreads-1)//nthreads, 1, 1)
    _kernel(c, a, b, np.int32(N), block=block, grid=grid)
    

for shape in [1024*128-1, 1024*1024+1]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    ms, _, _ = triton.testing.do_bench(lambda: a + b)
    print(f'torch: {ms} ms')

    for f in [kernel_compiled]:
        c = torch.zeros_like(a)
        BLOCK = 128
        f(a, b, c, N, BLOCK)
        assert(torch.allclose(c, a+b, atol=1e-3))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, N, BLOCK))
        print(f'kernel: {ms} ms')

