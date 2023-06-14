import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum

@slap.jit
def slap_kernel(a, b, c, M, N):
    for i in range(M):  #pragma parallel
        c[i] = sum(a[i,:N] * b[:N])
        
#@slap.jit  # TODO: need to support  1) 2D slicing 2) automatically add axis=1 for `block` version
def slap_kernel1(a, b, c, M, N, BM=8):
    for i in range(0, M, BM):  #pragma parallel
        c[i:i+BM] = sum(a[i:i+BM,:N] * b[:N], axis=1)
        
def torch_kernel(a, b, c, M, N):
    torch.mv(a, b, out=c)
    

for M, N in [(1024, 1024), (4096, 4096), (4096*4, 4096*4), (4096*16, 4096*8)]:
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.randn(M, device='cuda', dtype=torch.float32)
    c_ref = torch.randn(M, device='cuda', dtype=torch.float32)
    torch_kernel(a, b, c_ref, M, N)

    for f in (torch_kernel, slap_kernel):
        BLOCK = None
        f(a, b, c, M, N)
        assert(torch.allclose(c, c_ref, atol=1e-2))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, M, N))
        print(f'{f.__name__}: {ms:.4f} ms')
