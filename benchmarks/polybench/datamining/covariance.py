import torch
from appy import jit, max
import numpy as np
import numba as nb
from torch import zeros, empty, sum, maximum, add, exp, t, mm
from appy.utils import bench

torch.set_default_device('cuda')

def torch_kernel(M, float_n, data):
    mean = torch.mean(data, axis=0)
    data -= mean[None,:]
    cov = torch.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov

def mykernel(M, float_n, data, inner):
    mean = torch.mean(data, axis=0)
    data -= mean[None,:]
    cov = torch.zeros((M, M), dtype=data.dtype)
    inner(M, float_n, data, cov)
    return cov

@jit
def _mykernel(M, float_n, data, cov):
    #pragma parallel
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = sum(data[:float_n, i] * data[:float_n, j])
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]

@jit
def _mykernel_BN(M, float_n, data, cov, BN=256):
    #pragma parallel
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = 0.0
            for k in range(0, float_n, BN):
                cov[i, j] += sum(data[k:k+BN, i] * data[k:k+BN, j])
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]

@nb.jit(nopython=True, parallel=True, fastmath=True)
def numba_kernel(M, float_n, data):
    mean = np.sum(data, axis=0) / float_n
    data = data - mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in nb.prange(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov

def numpy_kernel(M, float_n, data):
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov

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
            b_ref = torch_kernel(M, N, a.clone())

            for f in (numpy_kernel, torch_kernel, _mykernel_BN, numba_kernel):
                if f.__name__.startswith('num'):
                    a_np = a.cpu().numpy()
                    ff = lambda: f(M, N, a_np)
                else:
                    if f.__name__.startswith('_'):
                        ff = lambda: mykernel(M, N, a, f)
                    else:
                        a_copy = a.clone()
                        ff = lambda: f(M, N, a_copy)
                    
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