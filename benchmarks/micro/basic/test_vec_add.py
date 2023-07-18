import torch
import numpy as np
import numba
from slap import jit
from slap.utils import bench
from torch import arange, zeros, empty

@jit(auto_block_slice=False)
def mykernel(a, b, c, N, BLOCK=256):
    #pragma parallel
    for i in range(0, N, BLOCK):  
        c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]

@jit
def mykernel1(a, b, c, N, BLOCK=256):  
    #pragma :N=>p,b(BLOCK)
    c[:N] = a[:N] + b[:N]

@numba.njit(cache=True)
def numba_kernel_seq(a, b, c, N, BLOCK=256):
    # #pragma parallel
    # for i in range(0, N, BLOCK):  
    #     c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]
    for i in range(0, N):  
        c[i] = a[i] + b[i]

@numba.njit(parallel=True, cache=True)
def numba_kernel_par(a, b, c, N, BLOCK=256):
    #pragma parallel
    for i in numba.prange(N):  
        c[i] = a[i] + b[i]

def torch_kernel(a, b, c, N):
    torch.add(a, b, out=c)

def numpy_kernel(a, b, c, N):
    np.add(a, b, out=c)
    
def test1():
    for device in ['cpu', 'cuda']:
        for dtype in [torch.float32, torch.float64]:
            for N in [1024*1024, 10*1024*1024, 100*1024*1024]:
                print(f'dtype: {dtype}, N: {N}')
                a = torch.randn(N, device=device, dtype=dtype)
                b = torch.randn(N, device=device, dtype=dtype)
                c_ref = torch.zeros_like(a)

                funcs = []

                if device == 'cpu':
                    a, b, c_ref = a.numpy(), b.numpy(), c_ref.numpy()
                    numpy_kernel(a, b, c_ref, N)
                    funcs = [numba_kernel_seq, numba_kernel_par, numpy_kernel]
                        
                else:
                    torch_kernel(a, b, c_ref, N)
                    funcs = [mykernel]
                        c = torch.zeros_like(a)                        
                        runner = lambda: f(a, b, c, N)
                        runner()
                        assert(torch.allclose(c, c_ref))

                for f in funcs:
                    c = torch.zeros_like(a)                        
                    runner = lambda: f(a, b, c, N)
                    runner()
                    assert(allclose(c, c_ref))
                    ms = bench(runner)
                    print(f'{f.__name__}: {ms:.4f} ms')
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                c_ref_np = c_ref.cpu().numpy()
                
                torch_kernel(a, b, c_ref, N)
                numpy_kernel(a_np, b_np, c_ref_np, N)
                
                for f in [numba_kernel_seq, numba_kernel_par, numpy_kernel, mykernel]:
                    c = torch.zeros_like(a)
                    c_np = c.cpu().numpy()
                    if f.__name__.startswith('numba') or f.__name__.startswith('numpy'):
                        runner = lambda: f(a_np, b_np, c_np, N)
                        runner()
                        assert(np.allclose(c_np, c_ref_np))
                    else:
                        runner = lambda: f(a, b, c, N)
                        runner()
                        assert(torch.allclose(c, c_ref))
                    ms = bench(runner)
                    print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()