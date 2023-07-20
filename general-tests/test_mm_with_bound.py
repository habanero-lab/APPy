import torch
from appy import jit, step
from appy.utils import bench, allclose
from torch import arange, zeros, empty, sum, float32

torch.set_default_device('cuda')

@jit
def mykernel1(a, b, c, M, N, K, BM=64, BN=128, BK=32):
    #pragma parallel
    for i in range(0, M, BM):  
        #pragma parallel
        for j in range(0, N, BN):
            i_BLOCK = step(i, BM, M)
            j_BLOCK = step(j, BN, N)
            acc = zeros([BM, BN], dtype=torch.float32)
            for k in range(0, K, BK):     
                k_BLOCK = step(k, BK, K)
                acc += a[i_BLOCK, k_BLOCK] @ b[k_BLOCK, j_BLOCK]
            c[i_BLOCK, j_BLOCK] = acc

def torch_kernel(a, b, c, M, N, K):
    torch.mm(a, b, out=c)
    
def test1():
    for dtype in [torch.float32]:
    #for dtype in [torch.float64]:
        for M, K, N in [(1000, 1100, 1200), (2000, 2300, 2600)]:
            print(f'M: {M}, N: {N}, K: {K}, dtype: {dtype}')
            a = torch.randn(M, K, device='cuda', dtype=dtype)
            b = torch.randn(K, N, device='cuda', dtype=dtype)
            
            c_ref = torch.randn(M, N, device='cuda', dtype=dtype)
            torch_kernel(a, b, c_ref, M, N, K)
            print(c_ref)

            for f in (torch_kernel, mykernel1):
                c = torch.randn(M, N, device='cuda', dtype=dtype)
                f(a, b, c, M, N, K)
                assert(allclose(c, c_ref))
                ms = bench(lambda: f(a, b, c, M, N, K))
                print(f'{f.__name__}: {ms:.4f} ms')
          

if __name__ == '__main__':
    test1()