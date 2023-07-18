import torch
from slap import jit
from slap.utils import bench
from torch import arange, zeros, empty, sum

def mykernel(a, N):
    b = torch.empty(1, device=a.device, dtype=a.dtype)
    _mykernel(a, b, N)
    return b

@jit(auto_block_slice=False)
def _mykernel(a, b, N, BLOCK=512):
    b.fill_(0)
    #pragma parallel reduction(b)
    for i in range(0, N, BLOCK):  
        b[0] += sum(a[i:i+BLOCK])

def _mykernel1(a, b, N, BLOCK=512):
    #pragma :N=>parallel,block(BLOCK),reduce(+:b)
    b[0] = sum(a[:N])

def torch_kernel(a, N):
    b = sum(a)
    return b
    
def test1():
    #for dtype in [torch.float16, torch.float32]:
    for dtype in [torch.float32, torch.float64]:
        for shape in [1024*1024, 10*1024*1024, 100*1024*1024]:
            N = shape
            print(f'N: {N}, dtype: {dtype}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b_ref = torch_kernel(a, N)

            for f in (torch_kernel, mykernel):
                b = f(a, N)
                assert(torch.allclose(b, b_ref, atol=0.1, rtol=0.05))
                ms = bench(lambda: f(a, N))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()