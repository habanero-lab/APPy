import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty

@slap.jit
def slap_kernel0(a, b, c, N, BLOCK):  
    for i in range(0, N, BLOCK):  #pragma parallel
        c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]

@slap.jit
def slap_kernel(a, b, c, N, BLOCK):  
    for i in range(N):  #pragma parallel block(256)
        c[i] = a[i] + b[i]

def torch_kernel(a, b, c, N, BLOCK=None):
    torch.add(a, b, out=c)
    
def test1():
    for dtype in [torch.float16, torch.float32, torch.float64]:
        for shape in [1024*128, 1024*1024, 1024*1024*2]:
            N = shape
            print(f'dtype: {dtype}, N: {N}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b = torch.randn(N, device='cuda', dtype=dtype)
            c_ref = torch.zeros_like(a)
            torch_kernel(a, b, c_ref, N)
            
            for f in [torch_kernel, slap_kernel0, slap_kernel]:
                BLOCK = 128 * 2
                c = torch.zeros_like(a)
                f(a, b, c, N, BLOCK)
                assert(torch.allclose(c, c_ref))
                ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, N, BLOCK))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()