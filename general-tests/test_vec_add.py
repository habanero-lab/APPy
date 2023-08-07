import torch
import numpy as np
import numba
from appy import jit, vidx
from appy.utils import bench
from torch import arange, zeros, empty

@jit
def mykernel(a, b, c, N, BLOCK=256):
    #pragma parallel
    for i in range(0, N, BLOCK):  
        vi = vidx(i, BLOCK, bound=N)
        c[vi] = a[vi] + b[vi]

@jit(dump_final_appy=1)
def mykernel_top(a, b, c, N, BLOCK=256):
    #pragma :N=>parallel,block(BLOCK)
    c[:N] = a[:N] + b[:N] 

def torch_kernel(a, b, c, N):
    torch.add(a, b, out=c)

def test1():
    for dtype in [torch.float16, torch.float32]:
        for N in [1024*1024, 10*1024*1024]:
            print(f'dtype: {dtype}, N: {N}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b = torch.randn(N, device='cuda', dtype=dtype)
            c_ref = torch.zeros_like(a)
            torch_kernel(a, b, c_ref, N)
            
            for f in [torch_kernel, mykernel_top]:
                c = torch.zeros_like(a)
                f(a, b, c, N)
                assert(torch.allclose(c, c_ref))
                ms = bench(lambda: f(a, b, c, N))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()