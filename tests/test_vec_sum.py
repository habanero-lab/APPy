import torch
import slap
import triton
import triton.language as tl

@triton.jit
def _triton_kernel(a, b, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK
#    idx = i + tl.arange(0, BLOCK)
    idx = i + tl.arange(0, 128)

    buf = tl.zeros([128], dtype=tl.float32)
    for _ in range(BLOCK//128):
        buf += tl.load(a+idx)
        idx+=128
    s = tl.sum(buf, axis=0)

    # a_block = tl.load(a+idx)
    # s = tl.sum(a_block, axis=0)
    tl.atomic_add(b, s)
    #tl.store(a+idx, a_block/s)

def triton_kernel(a, b, BLOCK):
    grid = (a.shape[0]//BLOCK,)
    fn = _triton_kernel[grid](a, b, BLOCK)
    print(fn.asm['ptx'])



#@slap.jit(tune=['BLOCK'])
def kernel(a, b, BLOCK):
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel reduction(+:b)
        s = torch.sum(a[i:i+BLOCK])
        b[0] += s 
        

for shape in [1024*128, 1024*1024]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    ms, _, _ = triton.testing.do_bench(lambda: torch.sum(a))
    print(f'torch: {ms} ms')

    for f in [triton_kernel, kernel]:
        b = torch.zeros(1, device='cuda', dtype=torch.float32)
        BLOCK = 128*8
        f(a, b, BLOCK)
        assert(torch.allclose(b, torch.sum(a)))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, BLOCK))
        print(f'kernel: {ms} ms')
