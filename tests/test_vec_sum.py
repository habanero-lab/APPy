import numpy as np
import torch
import triton
import triton.language as tl
from torch import arange, zeros, empty

#@slap.jit(tune=['BLOCK'])
def kernel(a, b, N, BLOCK):
    b[0] = 0
    for i in range(0, N, BLOCK):  #pragma parallel reduction(+:b)
        b[0] += torch.sum(a[i:i+BLOCK])

@triton.jit
def _kernel(a, b, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK
    _t0 = i + tl.arange(0, BLOCK)
    _t1 = tl.load(a+_t0)
    _t2 = tl.sum(_t1, axis=0)
    tl.atomic_add(b+0, _t2)

def kernel_compiled(a, b, N, BLOCK):
    b[0] = 0
    # Start parallel for loop
    nblocks = (N+BLOCK-1) // BLOCK
    _kernel[(nblocks,)](a, b, N, BLOCK)
    

for shape in [1024*128, 1024*1024, 10*1024*1024]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    ms, _, _ = triton.testing.do_bench(lambda: torch.sum(a))
    print(f'torch: {ms} ms')

    for f in [kernel, kernel_compiled]:
        b = torch.zeros(1, device='cuda', dtype=torch.float32)
        BLOCK = 128 * 4
        f(a, b, N, BLOCK)
        assert(torch.allclose(b, torch.sum(a), atol=1e-3))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, N, BLOCK))
        print(f'kernel: {ms} ms')
