import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum

def kernel_op(a, b):
    t0 = a * b[None,:]
    c = sum(t0, axis=1)
    return c

def kernel_unfused(a, b, c, t0, M, N):
    for i in range(M):
        for j in range(N):
            #pragma broadcast b(axis=0)
            t0[i,j] = a[i,j] * b[j]

    for i in range(M):
        for j in range(N):
            #pragma reduction(index=j, axis=1)
            c[i] += t0[i,j]

def kernel_unfused(a, b, c, t0, M, N):
    for i in range(M):
        for j in range(N):
            #pragma broadcast b(axis=0)
            t0 = a[i,j] * b[j]
            #pragma reduction(j, axis=1)
            c[i] += t0

def seq(a, b, c, M, N):
    #pragma parallel block
    for i in range(M):  
        #pragma block
        for j in range(N):  
            #pragma tensorfy c[i] += sum(a[i,j] * b[None,j], axis=1)
            c[i] += a[i,j] * b[j]


def seq(a, b, c, M, N):
    for i in range(0, M, BM):  #pragma parallel block
        acc = zeros([BM], device=a.device, dtype=a.dtype)
        for j in range(0, N, BN):  #pragama block
            acc += sum(a[i:i+BM,j:j+BN] * b[j:j+BN], axis=1)
        a[i] = acc

@slap.jit
def slap_kernel(a, b, c, M, N):
    for i in range(M):  #pragma parallel
        c[i] = sum(a[i,:N] * b[:N])

@slap.jit
def slap_kernel1(a, b, c, M, N, BN=256):
    for i in range(M):  #pragma parallel
        acc = zeros([BN], device=a.device, dtype=a.dtype)
        for j in range(0, N, BN):
            acc += a[i,j:j+BN] * b[j:j+BN]
        c[i] = sum(acc)

@slap.jit
def slap_kernel2(a, b, c, M, N, BM=8, BN=256):
    for i in range(0, M, BM):  #pragma parallel
        acc = zeros([BM, BN], device=a.device, dtype=a.dtype)
        for j in range(0, N, BN):
            acc += a[i:i+BM,j:j+BN] * b[j:j+BN][None,:]
        c[i:i+BM] = sum(acc, axis=1)

def torch_kernel(a, b, c, M, N):
    torch.mv(a, b, out=c)
    
def test1():
    for dtype in [torch.float16, torch.float32]:
    #for dtype in [torch.float64]:
        for M, N in [(1024, 1024), (4096, 4096), (4096*4, 4096*4)]:
            print(f'M: {M}, N: {N}')
            a = torch.randn(M, N, device='cuda', dtype=dtype)
            b = torch.randn(N, device='cuda', dtype=dtype)
            c = torch.randn(M, device='cuda', dtype=dtype)
            c_ref = torch.randn(M, device='cuda', dtype=dtype)
            torch_kernel(a, b, c_ref, M, N)

            for f in (torch_kernel, slap_kernel, slap_kernel1, slap_kernel2):
                BLOCK = None
                f(a, b, c, M, N)
                assert(torch.allclose(c, c_ref, atol=10, rtol=0.1))
                ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, M, N))
                print(f'{f.__name__}: {ms:.4f} ms')
            #exit(1)

if __name__ == '__main__':
    test1()