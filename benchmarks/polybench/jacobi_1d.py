import torch
from appy import jit, max, step, debug_barrier
import numpy as np
import numba as nb
from torch import zeros, empty, sum, maximum, add, exp, t, mm, sqrt, dot
from appy.utils import bench, allclose

def numpy_kernel(TSTEPS, N, A, B):
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
    return A, B

#@nb.jit(nopython=True, parallel=True, fastmath=True)
@nb.njit
def numba_kernel(TSTEPS, N, A, B):
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
    return A, B

def torch_kernel(TSTEPS, N, A, B):
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
    return A, B

def my_kernel(TSTEPS, N, A, B):
    for t in range(1, TSTEPS):
        _my_kernel(TSTEPS, N, A, B)
    return A, B

@jit
def _my_kernel(TSTEPS, N, A, B):
    #pragma parallel
    B[1:N-1] = 0.33333 * (A[:N-2] + A[1:N-1] + A[2:N])
    #pragma parallel
    A[1:N-1] = 0.33333 * (B[:N-2] + B[1:N-1] + B[2:N])


def test1():
    for dtype in [np.float32, np.float64]:
    #for dtype in [torch.float64]:
        
        #for (M, N) in [(800, 3200), (4000, 32000)]:        
        for (M, N) in [(4000, 32000), (4000, 320000), (4000, 3200000)]:
            print(f'M: {M}, N: {N}, dtype: {dtype}')
            a_np = np.random.randn(N).astype(dtype)
            b_np = np.random.randn(N).astype(dtype)
            a = torch.from_numpy(a_np).to('cuda')
            b = torch.from_numpy(b_np).to('cuda')

            c_ref, d_ref = torch_kernel(M, N, a.clone(), b.clone())
            c_np_ref, d_np_ref = numpy_kernel(M, N, a_np.copy(), b_np.copy())

            for f in (
                numpy_kernel, 
                numba_kernel,
                torch_kernel,
                my_kernel,                
                
                ):
                if f.__name__.startswith('num'):
                    ff = lambda: f(M, N, a_np.copy(), b_np.copy())
                    ref = d_np_ref
                else:                    
                    ff = lambda: f(M, N, a.clone(), b.clone())
                    ref = d_ref
                c, d = ff()
                assert allclose(d, ref, atol=1e-4), f.__name__
                
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()
