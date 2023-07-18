import torch
from slap import jit
from slap.utils import bench
from torch import arange, zeros, empty, sum, float32

torch.set_default_device('cuda')

@jit
def mykernel(a, b, c, M, N, K, BM=64, BN=64, BK=64):
    #pragma parallel
    for i in range(0, M, BM):  
        #pragma parallel
        for j in range(0, N, BN):  
            acc = zeros([BM, BN], dtype=torch.float32)
            for k in range(0, K, BK):     
                acc += a[i:i+BM, k:k+BK] @ b[k:k+BK, j:j+BN]
            c[i:i+BM, j:j+BN] = acc

def mykernel1(a, b, c, M, N, K, BM=64, BN=64, BK=64):
    #pragma :M=>p,b(BM) :N=>p,b(BN) :K=>b(BK),reduce(+:c)
    c[:M, :N] = a[:M, :K] @ b[:K, :N]

def mykernel1_expanded(a, b, c, M, N, K, BM=64, BN=64, BK=64):
    #pragma parallel
    for i0 in range(0, M, BM):  
        #pragma parallel
        for i1 in range(0, N, BN):
            c[i0:i0+BM, i1:i1+BN] = 0.0  # initialize this due to reduction sum
            for i2 in range(0, K, BK):
                c[i0:i0+BM, i1:i1+BN] += a[i0:i0+BM, i2:i2+BK] @ b[i2:i2+BK, i1:i1+BN]  # use += due to sum reduce

def mykernel2(a, b, c, M, N, K, BM=64, BN=64, BK=64):
    #pragma parallel
    for i in range(0, M, BM):  
        #pragma parallel
        for j in range(0, N, BN): 
            c[i:i+BM, j:j+BN] = a[i:i+BM,:] @ b[:,j:j+BN]
    

def torch_kernel(a, b, c, M, N, K):
    torch.mm(a, b, out=c)
    
def test1():
    for dtype in [torch.float16, torch.float32]:
    #for dtype in [torch.float64]:
        for M, K, N in [(1024, 1024, 1024), (4096, 4096, 4096), (4096*4, 4096*4, 4096*4)]:
            print(f'M: {M}, N: {N}, K: {K}, dtype: {dtype}')
            a = torch.randn(M, K, device='cuda', dtype=dtype)
            b = torch.randn(K, N, device='cuda', dtype=dtype)
            c = torch.randn(M, N, device='cuda', dtype=dtype)
            c_ref = torch.randn(M, N, device='cuda', dtype=dtype)
            torch_kernel(a, b, c_ref, M, N, K)

            for f in (torch_kernel, mykernel):
                f(a, b, c, M, N, K)
                assert(torch.allclose(c, c_ref, atol=1, rtol=0.5))
                ms = bench(lambda: f(a, b, c, M, N, K))
                print(f'{f.__name__}: {ms:.4f} ms')
          

if __name__ == '__main__':
    test1()