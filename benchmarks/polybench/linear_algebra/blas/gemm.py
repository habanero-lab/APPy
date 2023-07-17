import torch
from slap import jit, max
import numpy as np
import numba
from torch import zeros, empty, sum, maximum, add, exp, t, mm, sqrt, dot
from slap.utils import bench, allclose

torch.set_default_device('cuda')

def torch_kernel(A, B, C, alpha, beta, NI, NJ, NK):
    for i in range(NI):
        C[i,:NJ] *= beta
        
        for k in range(NK):
            C[i,:NJ] += alpha * A[i,k] * B[k,:NJ]

@jit
def mykernel(A, B, C, alpha, beta, NI, NJ, NK):
    #pragma parallel
    for i in range(NI):
        C[i,:NJ] *= beta
        
        for k in range(NK):
            C[i,:NJ] += alpha * A[i,k] * B[k,:NJ]

@numba.jit(nopython=True, parallel=True, cache=True)
def numba_kernel(A, B, C, alpha, beta, NI, NJ, NK):
    #pragma parallel
    for i in numba.prange(NI):
        C[i,:NJ] *= beta
        
        for k in range(NK):
            C[i,:NJ] += alpha * A[i,k] * B[k,:NJ]

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        
        for N in [1024, 2048, 4096]:
            print(f'N: {N}')
            a = torch.randn(N, N, dtype=dtype).to('cuda')
            b = torch.randn(N, N, dtype=dtype).to('cuda')
            c_ref = torch.randn(N, N, dtype=dtype).to('cuda')
            mykernel(a, b, c_ref, 1.0, 0.0, N, N, N)

            for f in (mykernel, numba_kernel, ):
                c = torch.randn(N, N, dtype=dtype).to('cuda')
                if f.__name__.startswith('numba'):
                    a, b, c = a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()
                    ff = lambda: f(a, b, c, 1.0, 0.0, N, N, N)
                else:
                    ff = lambda: f(a, b, c, 1.0, 0.0, N, N, N)
                
                ff()
                assert(allclose(c, c_ref))
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()
