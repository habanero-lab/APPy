import torch
from slap import jit, step
from slap.utils import bench
from torch import arange, zeros, empty

@jit
def mykernel0(a, b, c, N, BLOCK=256):
    #pragma parallel
    for i in range(0, N, BLOCK):  
        c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]

@jit
def mykernel1(a, b, c, N, BLOCK=256):  
    #pragma parallel block(BLOCK)
    for i in range(N):  
        c[i] = a[i] + b[i]

@jit
def mykernel2(a, b, c, N, BLOCK=256):  
    #pragma par_dim(:N:BLOCK)
    c[:N] = a[:N] + b[:N]

def torch_kernel(a, b, c, N):
    torch.add(a, b, out=c)
    
def test1():
    for dtype in [torch.float16, torch.float32]:
        for shape in [1024*128, 1024*1024, 1024*1024*2, 1024*1024*2+1]:
            N = shape
            print(f'dtype: {dtype}, N: {N}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b = torch.randn(N, device='cuda', dtype=dtype)
            c_ref = torch.zeros_like(a)
            torch_kernel(a, b, c_ref, N)
            
            for f in [torch_kernel, mykernel0]:
                c = torch.zeros_like(a)
                f(a, b, c, N)
                assert(torch.allclose(c, c_ref))
                ms = bench(lambda: f(a, b, c, N))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()