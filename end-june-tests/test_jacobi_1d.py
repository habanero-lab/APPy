import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum

def slap_kernel(a, N, BLOCK):
    b = torch.zeros(N-2, device=a.device, dtype=a.dtype)
    _slap_kernel(a, b, N, BLOCK)
    return b

@slap.jit
def _slap_kernel(a, b, N, BLOCK):
    for i in range(1, N-1, BLOCK):  #pragma parallel
        i = range(i, i+BLOCK)
        b[i-1] = (a[i-1] + a[i] + a[i+1]) / 3

def torch_kernel(a, N, BLOCK):
    b = (a[0:N-2] + a[1:N-1] + a[2:N]) / 3
    return b
    
def test1():
    for shape in [1024*128, 1024*1024, 40*1024*1024]:
        N = shape
        print(f'N: {N}')
        a = torch.randn(N, device='cuda', dtype=torch.float32)
        b_ref = torch_kernel(a, N, None)

        for f in (torch_kernel, slap_kernel):
            BLOCK = 128 * 4
            b = f(a, N, BLOCK)
            assert(torch.allclose(b, b_ref, atol=1e-2))
            ms, _, _ = triton.testing.do_bench(lambda: f(a, N, BLOCK))
            print(f'{f.__name__}: {ms:.4f} ms')
