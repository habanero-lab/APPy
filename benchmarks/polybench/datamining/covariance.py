import torch
from appy import jit, max
import numpy as np
import numba as nb
from torch import zeros, empty, sum, maximum, add, exp, t, mm
from appy.utils import bench, allclose

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

@jit(print_ptx=0)
def _mykernel(M, float_n, data, cov):
    #pragma parallel
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = sum(data[0:float_n, i] * data[0:float_n, j])
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]

@jit(print_ptx=0)
def _mykernel_one_block(M, float_n, data, cov):
    #pragma parallel num_warps(8)
    for i in range(M):        
        for j in range(i, M):
            rg = step(0, 2048, bound=float_n)
            cov[i, j] = sum(data[rg, i] * data[rg, j])
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]


@nb.jit(nopython=True, fastmath=True)
def numba_nopy_kernel(M, float_n, data):
    mean = np.sum(data, axis=0) / float_n
    data = data - mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov

@nb.jit(nopython=True, parallel=True, fastmath=True)
def numba_nopy_par_kernel(M, float_n, data):
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
    for dtype in [torch.float32, torch.float64]:
    #for dtype in [torch.float64]:
        #for M, N in [(1024, 1024), (1024*4, 1024*4), (1024*16, 1024*16)]:
        for M, N in [ (1200, 1400)]:
        #for M, N in [(1024, 256*2), (4096, 4096), (4096*4, 4096*4), (4096, 4096*8), (4096, 4096*16), (128, 4096*16), (256, 4096*16)]:
        #for M, N in [(8, 256)]:
            print(f'M: {M}, N: {N}, dtype: {dtype}')
            # M vars and N observations. Each row has M vars
            a = torch.randn(N, M, dtype=dtype).to('cuda')
            a_np = a.cpu().numpy()
            b_ref = torch_kernel(M, N, a.clone())
            b_np_ref = numpy_kernel(M, N, a_np.copy())

            for f in (numpy_kernel,  torch_kernel, _mykernel, _mykernel_one_block):
                if f.__name__.startswith('num'):                    
                    ff = lambda: f(M, N, a_np.copy())
                    ref = b_np_ref
                else:
                    ref = b_ref
                    if f.__name__.startswith('_'):
                        ff = lambda: mykernel(M, N, a.clone(), f)
                    else:
                        ff = lambda: f(M, N, a.clone())
                try:
                    b = ff()
                    assert allclose(b, ref, atol=1e-5),  f.__name__ + ' : validation error'
                    ms = bench(ff)
                    print(f'{f.__name__}: {ms:.4f} ms')
                except Exception as e:
                    print(e)
            

if __name__ == '__main__':
    test1()