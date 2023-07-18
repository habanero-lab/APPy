import numpy as np
import torch
import triton
import triton.language as tl
from torch import arange, zeros, empty

#@appy.jit
def kernel(a, b, c, N, BLOCK):  
    for i in range(0, N, BLOCK):  #pragma parallel
        ii = range(i, i+BLOCK)
        c[ii] = a[ii] + b[ii]

@triton.jit
def _kernel(a, b, c, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK
    ii = i + tl.arange(0, BLOCK)
    aii = tl.load(a+ii)
    bii = tl.load(b+ii)
    tl.store(c+ii, aii+bii)

def kernel_compiled(a, b, c, N, BLOCK):
    nblocks = (N+BLOCK-1) // BLOCK
    _kernel[(nblocks,)](a, b, c, N, BLOCK)
    

for shape in [1024*128, 1024*1024, 1024*1024*2]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    ms, _, _ = triton.testing.do_bench(lambda: a + b)
    print(f'torch: {ms} ms')


    for f in [kernel_compiled]:
        c = torch.zeros_like(a)
        BLOCK = 128 * 1
        f(a, b, c, N, BLOCK)
        assert(torch.allclose(c, a+b))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, N, BLOCK))
        print(f'kernel: {ms} ms')

