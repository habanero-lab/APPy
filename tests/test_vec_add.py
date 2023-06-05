import torch
import bmp
from bmp import parallel
import triton
import triton.language as tl

# @triton.jit
# def _triton_kernel(a, b, c, BLOCK: tl.constexpr):
#     i = tl.program_id(0) * BLOCK
#     idx = i + tl.arange(0, BLOCK)
#     tl.store(c+idx, tl.load(a+idx) + tl.load(b+idx))

# def triton_kernel(a, b, c, BLOCK):
#     grid = (a.shape[0]//BLOCK,)
#     fn = _triton_kernel[grid](a, b, c, BLOCK)
#     #print(fn.asm['ptx'])

#@bmp.jit
def kernel(a, b, c, BLOCK: parallel):
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel
        idx = range(i, i+BLOCK)
        c[idx] = a[idx] + b[idx]
        

for shape in [1024*128, 1024*1024]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    ms, _, _ = triton.testing.do_bench(lambda: a + b)
    print(f'torch: {ms} ms')


    for f in [kernel]:
        c = torch.zeros_like(a)
        BLOCK = 128
        f(a, b, c, BLOCK)
        assert(torch.allclose(c, a+b))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, BLOCK))
        print(f'kernel: {ms} ms')

