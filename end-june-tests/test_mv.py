import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum

@slap.jit
def slap_kernel(a, b, c, M, N, BLOCK=None):
    for i in range(M):  #pragma parallel
        c[i] = sum(a[i,:N] * b[:N])
        
def torch_kernel(a, b, c, M, N, BLOCK=None):
    torch.mv(a, b, out=c)
    

for M, N in [(1024, 1024), (4096, 4096), (4096*4, 4096*4)]:
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.randn(M, device='cuda', dtype=torch.float32)
    c_ref = torch.randn(M, device='cuda', dtype=torch.float32)
    torch_kernel(a, b, c_ref, M, N)

    for f in (torch_kernel, slap_kernel):
        BLOCK = 128 * 4
        f(a, b, c, M, N, BLOCK)
        assert(torch.allclose(c, c_ref, atol=1e-2))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, M, N, BLOCK))
        print(f'{f.__name__}: {ms:.4f} ms')
