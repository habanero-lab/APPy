import torch
from appy import jit, max
import numpy as np
import numba as nb
from torch import zeros, empty, sum, maximum, add, exp, t, mm
from appy.utils import bench, allclose

torch.set_default_device('cuda')

def torch_kernel(M, float_n, data):
    mean = torch.mean(data, axis=0)
    stddev = torch.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = torch.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]
    return corr

def mykernel(M, float_n, data, inner):
    mean = torch.mean(data, axis=0)
    stddev = torch.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = torch.eye(M, dtype=data.dtype)
    inner(M, float_n, data, corr)
    return corr

@jit
def _mykernel(M, float_n, data, corr):
    #pragma parallel
    for i in range(M - 1):
        for j in range(i+1, M):
            corr[i, j] = sum(data[:float_n, i] * data[:float_n, j])
            corr[j, i] = corr[i, j]

@jit
def _mykernel_BN(M, float_n, data, corr, BN=256):
    #pragma parallel
    for i in range(M - 1):
        for j in range(i+1, M):
            corr[i, j] = 0.0
            for k in range(0, float_n, BN):
                corr[i, j] += sum(data[k:k+BN, i] * data[k:k+BN, j])
            corr[j, i] = corr[i, j]

@nb.jit(nopython=True, parallel=True, fastmath=True)
def numba_nopy_par_kernel(M, float_n, data):
    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.sum((data - mean)**2, axis=0) / float_n)
    stddev[stddev <= 0.1] = 1.0
    
    data = data - mean
    
    data = data / (np.sqrt(float_n) * stddev).astype(np.float32)
    
    corr = np.eye(M, dtype=data.dtype)
    for i in nb.prange(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]

    return corr

def numpy_kernel(M, float_n, data):
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]

    return corr

def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        #for M, N in [(1024, 1024), (1024*4, 1024*4), (1024*16, 1024*16)]:
        for M, N in [(512, 512), (1024, 1024), (1200, 1280)]:
        #for M, N in [(1024, 256*2), (4096, 4096), (4096*4, 4096*4), (4096, 4096*8), (4096, 4096*16), (128, 4096*16), (256, 4096*16)]:
        #for M, N in [(8, 256)]:
            print(f'M: {M}, N: {N}')
            # M vars and N observations. Each row has M vars
            a = torch.randn(N, M, dtype=dtype).to('cuda')
            a_np = a.cpu().numpy()
            b_ref = torch_kernel(M, N, a.clone())

            for f in (_mykernel, numpy_kernel, numba_nopy_par_kernel, torch_kernel, _mykernel_BN):
                if f.__name__.startswith('num'):                    
                    ff = lambda: f(M, N, a_np.copy())
                else:
                    if f.__name__.startswith('_'):
                        ff = lambda: mykernel(M, N, a.clone(), f)
                    else:
                        ff = lambda: f(M, N, a.clone())
                    
                b = ff()
                assert(allclose(b, b_ref, atol=1e-3))
                ms = bench(ff)
                print(f'{f.__name__}: {ms:.4f} ms')
            

if __name__ == '__main__':
    test1()