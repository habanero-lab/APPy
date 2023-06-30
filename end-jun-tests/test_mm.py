import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum, float32

@slap.jit
def slap_kernel0(a, b, c, M, N, K, BM=64, BN=64, BK=32):
    for i in range(0, M, BM):  #pragma parallel
        for j in range(0, N, BN):  #pragma parallel
            acc = zeros([BM, BN], device=a.device, dtype=torch.float32)
            for k in range(0, K, BK):     
                acc += a[i:i+BM,k:k+BK] @ b[k:k+BK, j:j+BN]
            c[i:i+BM,j:j+BN] = acc

#@slap.jit
def slap_kernel(a, b, c, M, N, K, BM=64, BN=64, BK=32):
    for i in range(M):  #pragma parallel block(BM)
        for j in range(N):  #pragma parallel block(BN)
            acc: float32 = 0
            for k in range(K):  #pragma reduction(+:acc) block(BK)
                acc += a[i,k] * b[k,j]
            c[i,j] = acc


def torch_kernel(a, b, c, M, N, K):
    torch.mm(a, b, out=c)
    
def test1():
    for dtype in [torch.float16, torch.float32]:
    #for dtype in [torch.float64]:
        for M, N in [(1024, 1024), (4096, 4096), (4096*4, 4096*4)]:
            K = 64
            print(f'M: {M}, N: {N}, K: {K}')
            a = torch.randn(M, K, device='cuda', dtype=dtype)
            b = torch.randn(K, N, device='cuda', dtype=dtype)
            c = torch.randn(M, N, device='cuda', dtype=dtype)
            c_ref = torch.randn(M, N, device='cuda', dtype=dtype)
            torch_kernel(a, b, c_ref, M, N, K)

            for f in (torch_kernel, slap_kernel0):
                f(a, b, c, M, N, K)
                assert(torch.allclose(c, c_ref, atol=10, rtol=0.1))
                ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, M, N, K))
                print(f'{f.__name__}: {ms:.4f} ms')
            #exit(1)

if __name__ == '__main__':
    test1()