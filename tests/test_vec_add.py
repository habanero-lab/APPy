import numpy as np
import torch
import triton
from slap import parallel, prange
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule

from torch import arange, zeros, empty

#@slap.jit
def kernel(a, b, c, N, BLOCK: parallel):
    for i in range(0, N, BLOCK):
        ii = range(i, i+BLOCK)
        c[ii] = a[ii] + b[ii]

def kernel_compiled(a, b, c, N, BLOCK: parallel):
    if not hasattr(kernel_compiled, 'cached'):
        kernel_compiled.cached = {}

    threadblock = (128, 1, 1)
    assert BLOCK % threadblock[0] == 0
    sig = (threadblock, a, b, c, BLOCK)
    
    if sig not in kernel_compiled.cached:
        mod = SourceModule("""
            __global__ void _kernel(float *c, float *a, float *b, int N, int BLOCK) {
                int i = blockIdx.x * BLOCK;
                int ii = i + threadIdx.x;
                while (ii < i+BLOCK) {
                    c[ii] = a[ii] + b[ii];
                    ii += blockDim.x;
                }
            }
        """, options=['-O3'])
      
        _kernel = mod.get_function("_kernel")
        kernel_compiled.cached[sig] = _kernel

    _kernel = kernel_compiled.cached[sig]
    threadblock = sig[0]
    grid = (N // BLOCK, 1, 1)
    _kernel(c, a, b, np.int32(N), np.int32(BLOCK), block=threadblock, grid=grid)
    

for shape in [1024*128, 1024*1024, 1024*1024*2]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    ms, _, _ = triton.testing.do_bench(lambda: a + b)
    print(f'torch: {ms} ms')


    for f in [kernel, kernel_compiled]:
        c = torch.zeros_like(a)
        BLOCK = 128 * 1
        f(a, b, c, N, BLOCK)
        assert(torch.allclose(c, a+b))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, N, BLOCK))
        print(f'kernel: {ms} ms')

