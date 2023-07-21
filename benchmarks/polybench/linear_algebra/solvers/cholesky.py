import torch
from appy import jit, max, step, debug_barrier
import numpy as np
import numba as nb
from torch import zeros, empty, sum, maximum, add, exp, t, mm, sqrt, dot
from appy.utils import bench, allclose

torch.set_default_device('cuda')

'''
i = 0
    A[0,0] = sqrt(A[0,0])
i = 1
    j = 0
        A[1,0] -= 0
        A[1,0] /= A[0,0]
    A[1,1] = 
i = 2
    j = 0
        A[2,0] -= 0
        A[2,0] /= A[0,0]
    j = 1
        A[2,1] -= dot(A[2,0:1, A[1,0:1])
        A[2,1] /= A[1,1]
'''
def numpy_kernel(A, N):
    for i in range(N):
        # j < i
        for j in range(0, i):
            A[i,j] -= np.dot( A[i,0:j], A[j, 0:j] )
            A[i,j] /= A[j,j]

        # i == j case
        #A[i,i] -= np.dot( A[i,0:i], A[i,0:i] )
        A[i,i] = np.sqrt(A[i,i])
    return A

@nb.jit(nopython=True, parallel=True, fastmath=True)
def numba_kernel(A, N):
    for i in nb.prange(N):
        # j < i
        for j in range(0, i):
            A[i,j] -= np.dot( A[i,0:j], A[j, 0:j] )
            A[i,j] /= A[j,j]

        # i == j case
        #A[i,i] -= np.dot( A[i,0:i], A[i,0:i] )
        A[i,i] = np.sqrt(A[i,i])
    return A

def torch_kernel(A, N):
    for i in range(N):
        # j < i
        for j in range(0, i):
            A[i,j] -= dot( A[i,0:j], A[j, 0:j] )
            A[i,j] /= A[j,j]

        # i == j case
        #A[i,i] -= dot( A[i,0:i], A[i,0:i] )
        A[i,i] = sqrt(A[i,i])
    return A

@jit
def mykernel(A, N, BLOCK=128):
    #pragma parallel
    for z in range(1):
        for i in range(N):
            # j < i
            for j in range(0, i):
                s = 0.0
                for k in range(0, j, BLOCK):
                    kk = step(k, BLOCK, bound=j)
                    s += sum( A[i,kk] * A[j,kk] )
                
                A[i,j] -= s
                A[i,j] /= A[j,j]

            debug_barrier()

            # i == j case
            s = 0.0
            for k in range(0, i, BLOCK):
                kk = step(k, BLOCK, bound=i)
                s += sum( A[i,kk] * A[j,kk] )
        
            A[i,i] -= s
            A[i,i] = sqrt(A[i,i])
    return A

@jit
def mykernel1(A, N, BLOCK=128):
    
        #for z in range(1):
        #pragma parallel
        for i in range(N):
            # j < i
            for j in range(0, i):
                offset = step(0, N, bound=j)
                s = sum(A[i, offset] * A[j, offset])
                A[i,j] -= s
                A[i,j] /= A[j,j]
                debug_barrier()
            
            # i == j case
            debug_barrier()
            offset = step(0, N, bound=i)
            
            #s = sum(A[i, offset] * A[i, offset])
            #A[i,i] -= s
            A[i,i] = sqrt(A[i,i])
        return A

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        
        for N in [128, 256, 512]:
        
        #for M, N in [(8, 256)]:
            print(f'N: {N}')
            a = torch.randn(N, N, dtype=dtype).to('cuda')
            a = a @ a.T / N
            a_np = a.cpu().numpy()
            b_ref = torch_kernel(a.clone(), N)

            for f in (
                numpy_kernel, 
                numba_kernel,
                mykernel1, 
                #torch_kernel
                ):
                if f.__name__.startswith('num'):
                    ff = lambda: f(a_np.copy(), N)
                else:
                    
                    ff = lambda: f(a.clone(), N)
                
                b = ff()
                assert(allclose(b, b_ref))
                
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()
