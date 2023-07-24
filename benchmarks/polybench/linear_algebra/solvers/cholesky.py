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
        A[i,i] -= np.dot( A[i,0:i], A[i,0:i] )
        #A[i,i] = np.sqrt(A[i,i])
    return A

#@nb.jit(nopython=True, parallel=True, fastmath=True)
@nb.njit
def numba_kernel(A, N):
    for i in range(N):
        # j < i
        for j in range(0, i):
            A[i,j] -= np.dot( A[i,0:j], A[j, 0:j] )
            A[i,j] /= A[j,j]

        # i == j case
        A[i,i] -= np.dot( A[i,0:i], A[i,0:i] )
        #A[i,i] = np.sqrt(A[i,i])
    return A

def torch_kernel(A, N):
    for i in range(N):
        # j < i
        for j in range(0, i):
            A[i,j] -= dot( A[i,0:j], A[j, 0:j] )
            A[i,j] /= A[j,j]

        # i == j case
        A[i,i] -= dot( A[i,0:i], A[i,0:i] )
        #A[i,i] = sqrt(A[i,i])
    return A

@jit
def mykernel(A, N, BLOCK=256):
    #pragma parallel
    for _ in range(1):
        for i in range(N):
            # j < i
            for j in range(0, i):
                s = sum( A[i,:j] * A[j,:j] )
                A[i,j] -= s
                A[i,j] /= A[j,j]
                debug_barrier()

            # i == j case
            s1 = sum( A[i,:i] * A[i,:i] )
            A[i,i] -= s1
            #A[i,i] = sqrt(A[i,i])
            debug_barrier()
            
    return A

@jit
def mykernel_one_block(A, N, BLOCK=4096):
    #pragma parallel num_warps(16)
    for _ in range(1):
        for i in range(N):
            # j < i
            for j in range(0, i):
                jj = step(0, BLOCK, j)
                s = sum( A[i,jj] * A[j,jj] )
                A[i,j] -= s
                A[i,j] /= A[j,j]
                debug_barrier()

            # i == j case
            ii = step(0, BLOCK, i)
            s1 = sum( A[i,ii] * A[i,ii] )
            A[i,i] -= s1
            #A[i,i] = sqrt(A[i,i])
            debug_barrier()
            
    return A

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        
        #for N in [256, 512, 1024, 2000]:
        for N in [4000]:        
            print(f'N: {N}')
            a = torch.randn(N, N, dtype=dtype).to('cuda')
            a = a @ a.T / N * 2
            a_np = a.cpu().numpy()
            b_ref = mykernel_one_block(a.clone(), N)
            b_np_ref = numpy_kernel(a_np.copy(), N)

            for f in (
                mykernel_one_block,
                numpy_kernel, 
                numba_kernel,
                    mykernel,
                    
                    torch_kernel,
                ):
                if f.__name__.startswith('num'):
                    ff = lambda: f(a_np.copy(), N)
                    ref = b_np_ref
                else:                    
                    ff = lambda: f(a.clone(), N)
                    ref = b_ref
                b = ff()
                assert allclose(b, ref), f.__name__
                
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()
