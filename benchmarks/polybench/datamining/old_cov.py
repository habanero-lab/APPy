import torch
from appy import jit, max
import numpy as np
import numba
from torch import zeros, empty, sum, maximum, add, exp, t, mm
from appy.utils import bench

torch.set_default_device('cuda')

def torch_kernel(M, float_n, data):
    mean = np.mean(data, axis=0)
    data -= mean
    cov = zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov

def mykernel(M, float_n, data):
    mean = np.mean(data, axis=0)
    data -= mean
    cov = zeros((M, M), dtype=data.dtype)
    #pragma parallel
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = np.sum(data[:float_n, i] * data[:float_n, j])
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]
    
    return cov

def mykernel(a, inner):
    n_vars, n_obs = a.shape
    new_a = torch.empty_like(a)
    b = torch.empty([n_vars, n_vars], dtype=a.dtype)
    inner(a, new_a, b, n_vars, n_obs)
    return b

def _torch_kernel(a, new_a, b, M, N):
    #pragma parallel 
    for i in range(M):  
        mean = sum(a[i,:N])
        mean /= N
        new_a[i,:N] = a[i,:N] - mean

    #pragma parallel
    for i in range(M):  
        for j in range(i, M):  
            b[i,j] = sum(new_a[i,:N] * new_a[j,:N])
            b[i,j] /= N - 1
            b[j,i] = b[i,j]

@jit
def _mykernel(a, new_a, b, M, N, BN=256):
    #pragma parallel 
    for i in range(M):  
        mean = sum(a[i,:N])
        mean /= N
        new_a[i,:N] = a[i,:N] - mean

    #pragma parallel
    for i in range(M):  
        for j in range(i, M):
            b[i,j] = sum(new_a[i,:N] * new_a[j,:N])
            b[i,j] /= N - 1
            b[j,i] = b[i,j]

@jit
def _mykernel_BN(a, new_a, b, M, N, BN=256):
    #pragma parallel
    for i in range(M): 
        mean = 0.0
        for j in range(0, N, BN):
            mean += sum(a[i,j:j+BN])
        mean /= N
        for j in range(0, N, BN):
            new_a[i,j:j+BN] = a[i,j:j+BN] - mean

    #pragma parallel
    for i in range(M):  
        for j in range(i, M):
            b[i,j] = 0.0
            for k in range(0, N, BN):
                b[i,j] += sum(new_a[i,k:k+BN] * new_a[j,k:k+BN])
            
            b[i,j] /= N - 1
            b[j,i] = b[i,j]      

def numba_kernel(a):
    n_vars, n_obs = a.shape
    new_a = np.empty_like(a)
    b = np.empty([n_vars, n_vars], dtype=a.dtype)
    _numba_kernel(a, new_a, b, n_vars, n_obs)
    return b

@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_kernel(a, new_a, b, M, N):
    for i in numba.prange(M):  
        mean = np.sum(a[i,:N])
        mean /= N
        new_a[i,:N] = a[i,:N] - mean

    for i in numba.prange(M):  
        for j in range(i, M):  
            b[i,j] = np.sum(new_a[i,:N] * new_a[j,:N])
            b[i,j] /= N - 1
            b[j,i] = b[i,j]


def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        #for M, N in [(1024, 1024), (1024*4, 1024*4), (1024*16, 1024*16)]:
        for M, N in [(512, 512), (1024, 1024), (1024*4, 1024), (1024*4, 1024*2), (1024*4, 1024*4)]:
        #for M, N in [(1024, 256*2), (4096, 4096), (4096*4, 4096*4), (4096, 4096*8), (4096, 4096*16), (128, 4096*16), (256, 4096*16)]:
        #for M, N in [(8, 256)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, dtype=dtype).to('cuda')
            b_ref = mykernel(a, _mykernel)

            for f in (_mykernel, _mykernel_BN, numba_kernel):
                if f.__name__.startswith('numba'):
                    a_np = a.cpu().numpy()
                    ff = lambda: f(a_np)
                else:
                    if f.__name__.startswith('_'):
                        ff = lambda: mykernel(a, f)
                    else:
                        ff = lambda: f(a)
                    
                b = ff()
                if isinstance(b, np.ndarray):
                    b = torch.from_numpy(b).to('cuda')

                if not torch.allclose(b, b_ref, atol=0.1, rtol=0.05):
                    print(b)
                    print(b_ref)
                assert(torch.allclose(b, b_ref, atol=0.1, rtol=0.05))
                #ms, _, _ = triton.testing.do_bench(ff)
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()