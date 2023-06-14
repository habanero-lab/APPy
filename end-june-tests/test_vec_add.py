import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty

@slap.jit
def slap_kernel(a, b, c, N, BLOCK):  
    for i in range(0, N, BLOCK):  #pragma parallel
        c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]

@slap.jit(dump_code=1)
def slap_kernel1(a, b, c, N, BLOCK):  
    for i in range(N):  #pragma parallel block(128)
        c[i] = a[i] + b[i]

@triton.jit
def _kernel(a, b, c, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK
    _t0 = i + tl.arange(0, BLOCK)
    _t1 = tl.load(a+_t0)
    _t2 = tl.load(b+_t0)
    tl.store(c+_t0, _t1+_t2)

def triton_kernel(a, b, c, N, BLOCK):
    nblocks = (N+BLOCK-1) // BLOCK
    _kernel[(nblocks,)](a, b, c, N, BLOCK)
    
def test1():
    for shape in [1024*128, 1024*1024, 1024*1024*2]:
        N = shape
        print(f'N: {N}')
        a = torch.randn(N, device='cuda', dtype=torch.float32)
        b = torch.randn(N, device='cuda', dtype=torch.float32)
        ms, _, _ = triton.testing.do_bench(lambda: a + b)
        print(f'torch: {ms} ms')

        #slap_kernel = slap.jit(kernel)

        for f in [slap_kernel, slap_kernel1]:
            c = torch.zeros_like(a)
            BLOCK = 128 * 1
            f(a, b, c, N, BLOCK)
            assert(torch.allclose(c, a+b))
            ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, N, BLOCK))
            print(f'{f.__name__}: {ms:.4f} ms')

