import numpy as np
import torch
import triton
from pathlib import Path
from slap import parallel, prange
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule

from torch import arange, zeros, empty

#@slap.jit(tune=['BLOCK'])
def kernel(a, b, N, BLOCK: parallel):
    for i in range(0, N, BLOCK):  #pragma parallel reduction(+:b)
        b[0] += torch.sum(a[i:i+BLOCK])


def kernel_compiled(a, b, N, BLOCK: parallel):
    if not hasattr(kernel_compiled, 'cached'):
        kernel_compiled.cached = {}

    threadblock = (128, 1, 1)
    assert BLOCK % threadblock[0] == 0
    sig = (threadblock, a, b, BLOCK)
    
    if sig not in kernel_compiled.cached:
        src = Path('/home/tong/projects/SLAP/utils.cu').read_text()
        src += '''
        __global__ void _kernel(float *a, float *b, int N, int BLOCK) {
            int i = blockIdx.x * BLOCK;
            int ii = i + threadIdx.x;
            float sum = 0;
            while (ii < i+BLOCK) {
                sum += a[ii];
                ii += blockDim.x;
            }
            sum = blockReduceSum(sum);

            // How to know here just one thread should act?
            if (threadIdx.x == 0) {
                atomicAdd(b, sum);
            }
        }
        '''
        mod = SourceModule(src, options=['-O3'])
      
        _kernel = mod.get_function("_kernel")
        kernel_compiled.cached[sig] = _kernel

    _kernel = kernel_compiled.cached[sig]
    threadblock = sig[0]
    grid = (N // BLOCK, 1, 1)
    _kernel(a, b, np.int32(N), np.int32(BLOCK), block=threadblock, grid=grid)
     

for shape in [1024*128, 1024*1024]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    ms, _, _ = triton.testing.do_bench(lambda: torch.sum(a))
    print(f'torch: {ms} ms')

    for f in [kernel, kernel_compiled]:
        b = torch.zeros(1, device='cuda', dtype=torch.float32)
        BLOCK = 128
        f(a, b, N, BLOCK)
        assert(torch.allclose(b, torch.sum(a)))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, N, BLOCK))
        print(f'kernel: {ms} ms')
