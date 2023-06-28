import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum

def slap_kernel(a, N):
    b = torch.empty(N-2, device=a.device, dtype=a.dtype)
    _slap_kernel(a, b, N)
    return b

@slap.jit
def _slap_kernel(a, b, N, BLOCK=512):
    for i in range(0, N-2):  #pragma parallel block(BLOCK)
        b[i] = (a[i] + a[i+1] + a[i+2]) / 3

def torch_kernel(a, N):
    b = (a[0:N-2] + a[1:N-1] + a[2:N]) / 3
    return b
    
def test1():
    for shape in [1024*128, 1024*1024, 40*1024*1024]:
        N = shape
        print(f'N: {N}')
        a = torch.randn(N, device='cuda', dtype=torch.float32)
        b_ref = torch_kernel(a, N)

        for f in (torch_kernel, slap_kernel):
            b = f(a, N)
            assert(torch.allclose(b, b_ref, atol=1e-2))
            ms, _, _ = triton.testing.do_bench(lambda: f(a, N))
            print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()