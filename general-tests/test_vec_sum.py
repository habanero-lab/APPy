import torch
from appy import jit, step
from appy.utils import bench
from torch import arange, zeros, empty, sum

@jit
def mykernel(a, b, N, BLOCK=512):
    b.fill_(0)
    #pragma parallel reduction(b)
    for i in range(0, N, BLOCK):  
        b[0] += sum(a[i:i+BLOCK])

@jit
def mykernel1(a, b, N, BLOCK=512):
    b.fill_(0)
    #pragma parallel reduction(b)
    for i in range(0, N, BLOCK):  
        i_BLOCK = step(i, BLOCK)
        b[0] += sum(a[i_BLOCK])

def torch_kernel(a, b, N):
    sum(a, dim=0, out=b)
    
def test1():
    #for dtype in [torch.float16, torch.float32]:
    for dtype in [torch.float32, torch.float64]:
        for shape in [1024*1024, 10*1024*1024, 100*1024*1024]:
            N = shape
            print(f'N: {N}, dtype: {dtype}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b_ref = torch.empty(1, device='cuda', dtype=dtype)
            torch_kernel(a, b_ref, N)

            for f in (torch_kernel, mykernel, mykernel1):
                b = torch.empty(1, device='cuda', dtype=dtype)
                f(a, b, N)
                assert(torch.allclose(b, b_ref, atol=0.1, rtol=0.05))
                ms = bench(lambda: f(a, b, N))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()