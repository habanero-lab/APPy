import torch
from appy import jit, max, step
import numpy as np
import numba
from torch import zeros, empty, sum, maximum, add, exp, t, mm, sqrt, dot
from appy.utils import bench

torch.set_default_device('cuda')

def torch_kernel(A, N):
    for i in range(N):
        # j < i
        for j in range(0, i):
            A[i,j] -= dot( A[i,0:j], A[j, 0:j] )
            A[i, j] /= A[j, j]

        # i == j case
        #print(A[i,i])
        A[i,i] -= dot( A[i,0:i], A[i,0:i] )
        A[i,i] = sqrt(A[i,i])

@jit
def mykernel(A, N, BLOCK=128):
    #pragma parallel
    for i in range(N):
        # j < i
        for j in range(0, i):
            s = 0.0
            for k in range(0, j, BLOCK):
                kk = step(k, BLOCK, bound=j)
                s += sum( A[i,kk] * A[j,kk] )
            
            A[i,j] -= s
            A[i,j] /= A[j,j]

        # i == j case
        s = 0.0
        for k in range(0, i, BLOCK):
            kk = step(k, BLOCK, bound=i)
            s += sum( A[i,kk] * A[i,kk] )
        A[i,i] -= s
        A[i,i] = sqrt(A[i,i])


def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        
        for N in [256, 512, 1024]:
        
        #for M, N in [(8, 256)]:
            print(f'N: {N}')
            a = torch.randn(N, N, dtype=dtype).to('cuda')
            a = a @ a.T / N
            a_ref = a.clone()
            mykernel(a_ref, N)

            for f in (torch_kernel, mykernel):
                if f.__name__.startswith('numba'):
                    b = a.cpu().numpy()
                    ff = lambda: f(b, N)
                else:
                    b = a.clone()
                    ff = lambda: f(b, N)
                
                ff()
                if isinstance(b, np.ndarray):
                    b = torch.from_numpy(b).to('cuda')

                if not torch.allclose(b, a_ref, atol=0.1, rtol=0.05):
                    print(b)
                    print(a_ref)
                assert(torch.allclose(b, a_ref, atol=0.1, rtol=0.05))
                #ms, _, _ = triton.testing.do_bench(ff)
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()
