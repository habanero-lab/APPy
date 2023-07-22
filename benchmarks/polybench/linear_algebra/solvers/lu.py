import torch
from appy import jit, max, step, debug_barrier
import numpy as np
import numba as nb
from torch import zeros, empty, sum, maximum, add, exp, t, mm, sqrt, dot
from appy.utils import bench, allclose

torch.set_default_device('cuda')

def numpy_kernel(A, N):
    for i in range(N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[:j, j])
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= np.dot(A[i, :i], A[:i, j])
    return A

#@nb.jit(nopython=True, parallel=True, fastmath=True)
@nb.njit
def numba_kernel(A, N):
    for i in range(N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[:j, j])
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= np.dot(A[i, :i], A[:i, j])
    return A

def torch_kernel(A, N):
    for i in range(N):
        for j in range(i):
            A[i, j] -= torch.dot(A[i, :j], A[:j, j])
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= torch.dot(A[i, :i], A[:i, j])
    return A

@jit
def mykernel(A, N):
    #pragma parallel
    for _ in range(1):
        for i in range(N):
            for j in range(i):
                s0 = sum(A[i, :j] * A[:j, j])
                A[i, j] -= s0
                A[i, j] /= A[j, j]
                debug_barrier()
            for j in range(i, N):
                s1 = sum(A[i, :i] * A[:i, j])
                A[i, j] -= s1
                debug_barrier()
    return A

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        
        for N in [256, 512, 1024, 2000]:
        
        #for M, N in [(8, 256)]:
            print(f'N: {N}')
            a = torch.randn(N, N, dtype=dtype).to('cuda')
            a = a @ a.T / N * 2
            a_np = a.cpu().numpy()
            b_ref = torch_kernel(a.clone(), N)
            b_np_ref = numpy_kernel(a_np.copy(), N)

            for f in (
                numpy_kernel, 
                #numba_kernel,            
                mykernel,
                torch_kernel
                ):
                if f.__name__.startswith('num'):
                    ff = lambda: f(a_np.copy(), N)
                    ref = b_np_ref
                else:                    
                    ff = lambda: f(a.clone(), N)
                    ref = b_ref
                b = ff()
                assert allclose(b, ref, atol=1e-2), f.__name__
                
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()
