import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum

def slap_kernel(a, N):
    b = torch.zeros(1, device=a.device, dtype=a.dtype)
    _slap_kernel(a, b, N)
    return b

@slap.jit
def _slap_kernel(a, b, N, BLOCK=512):
    #pragma parallel reduction(b)
    for i in range(0, N, BLOCK):  
        b[0] += sum(a[i:i+BLOCK])

def _slap_kernel(a, b, N, BLOCK=512):
    #pragma parallel reduction(b)
    for i in range(N):  
        b[0] += sum(a[i])

def torch_kernel(a, N):
    b = sum(a)
    return b
    
def test1():
    for dtype in [torch.float16, torch.float32]:
        for shape in [1024*128, 1024*1024, 10*1024*1024, 10*1024*1024+1]:
            N = shape
            print(f'N: {N}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b_ref = torch_kernel(a, N)

            for f in (torch_kernel, slap_kernel):
                b = f(a, N)
                
                assert(torch.allclose(b, b_ref, atol=0.1, rtol=0.1))
                ms, _, _ = triton.testing.do_bench(lambda: f(a, N))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()