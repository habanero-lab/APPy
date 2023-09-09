import torch
import appy
from appy.utils import *

#@appy.jit(configs=appy.get_matmul_configs('BM', 'BN', 'BK'))
def kernel(A, B, C, BM=32, BN=32, BK=32):
    M, K = A.shape
    K, N = B.shape
    #pragma parallel
    for m in range(0, M, BM):
        #pragma parallel
        for n in range(0, N, BN):
            vm = appy.vidx(m, BM, bound=M)
            vn = appy.vidx(n, BN, bound=N)
            C[vm, vn] = 0
            for k in range(0, K, BK):
                vk = appy.vidx(k, BK, bound=K)
                C[vm, vn] += A[vm, vk] @ B[vk, vn] 


@appy.jit(configs=appy.get_matmul_configs('BM', 'BN', 'BK'))
def kernel1(A, B, C, BM=64, BN=64, BK=32):
    M, K = A.shape
    K, N = B.shape
    #pragma parallel
    for m in range(0, M, BM):
        #pragma parallel
        for n in range(0, N, BN):
            vm = appy.vidx(m, BM, bound=M)
            vn = appy.vidx(n, BN, bound=N)
            acc = torch.zeros([BM, BN], dtype=torch.float32)
            for k in range(0, K, BK):
                vk = appy.vidx(k, BK, bound=K)
                acc += A[vm, vk] @ B[vk, vn]
            C[vm, vn] = acc 

@appy.jit(configs=appy.get_matmul_configs('BM', 'BN', 'BK'))
def kernel_op(A, B, C):
    M, K = A.shape
    K, N = B.shape
    #pragma :M=>parallel,block(BM) :N=>parallel,block(BN) :K=>block(BK)
    C[:M, :N] = A[:M, :K] @ B[:K, :N]

# def kernel(a, b, c, Bi, Bj, Bk):
#     for i in range(0, a.shape[0], Bi):  #pragma parallel
#         for j in range(0, b.shape[-1], Bj):  #pragma parallel
#             c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
#             for k in range(0, a.shape[-1]):
#                 c_block[:, :] += a[i:i+Bi, k] * b[k, j:j+Bj]
#             c[i:i+Bi, j:j+Bj] = c_block


# def kernel(a, b, c, Bi:parallel, Bj:parallel, Bk):
#     for i in range(0, a.shape[0], Bi):  #pragma parallel
#         for j in range(0, b.shape[-1], Bj):  #pragma parallel
#             c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
#             a_block:shared = torch.zeros([Bi, Bk], device=a.device, dtype=a.dtype)
#             b_block:shared = torch.zeros([Bk, Bj], device=a.device, dtype=a.dtype)
#             for k in range(0, a.shape[-1], Bk):
#                 # Load data to shared memory
#                 appy.syncthreads()
#                 a_block[:,:] = a[i:i+Bi, k:k+Bj]
#                 b_block[:,:] = b[k:k+Bi, j:j+Bj]
#                 appy.syncthreads()

#                 for kk in range(k, k+Bk):
#                     c_block[:, :] += a_block[:Bi, kk] * b_block[kk, :Bj]
                
#             c[i:i+Bi, j:j+Bj] = c_block


# #@appy.jit(tune=['Bi', 'Bj', 'Bk'])
# def kernel(a, b, c, Bi, Bj, Bk):
#     for i in range(0, a.shape[0], Bi):  #pragma parallel
#         for j in range(0, b.shape[-1], Bj):  #pragma parallel
#             c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
#             for k in range(0, a.shape[-1], Bk):
#                 c_block[:, :] += a[i:i+Bi, k:k+Bk] @ b[k:k+Bk, j:j+Bj]
#             c[i:i+Bi, j:j+Bj] = c_block

def torch_kernel(A, B, C):
    torch.matmul(A, B, out=C)

for M, N, K in [(1024, 1024, 1024), (4096, 4096, 4096), (1200, 1300, 1400)]:
    print(f'M: {M}, N: {N}, K: {K}')
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    c_ref = a @ b

    for f in [torch_kernel, kernel_op]:
        c = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        
        f(a, b, c)
        assert(allclose(c_ref, c, atol=1e-3))
        print(f'{f.__name__} time: {bench(lambda: f(a, b, c))} ms')
