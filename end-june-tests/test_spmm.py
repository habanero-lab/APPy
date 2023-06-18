import torch
import triton
import slap
from torch import arange, zeros, empty, sum

@slap.jit
def slap_kernel0(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty([M, N], device='cuda', dtype=torch.float32)
    a_rowptrs, a_cols, a_vals = a.crow_indices(), a.col_indices(), a.values()
    BN = N  
    #pragma parallel const(M, N, BN)
    for i in range(M):  
        #pragma parallel 
        for j in range(0, N, BN):  
            acc = zeros(BN, device='cuda', dtype=torch.float32)
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                acc += a_ik * b[ks,j:j+BN]
            c[i,j:j+BN] = acc
    return c

@slap.jit
def _slap_kernel0(a_rowptrs, a_cols, a_vals, b, c, M, K, N, BN):
    for i in range(M):  #pragma parallel 
        for j in range(0, N, BN):  #pragma parallel 
            acc = zeros(BN, device='cuda', dtype=torch.float32)
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                acc += a_ik * b[ks,j:j+BN]
            c[i,j:j+BN] = acc

@slap.jit
def slap_kernel(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty([M, N], device='cuda', dtype=torch.float32)
    a_rowptrs, a_cols, a_vals = a.crow_indices(), a.col_indices(), a.values()
    #pragma parallel 
    for i in range(M):  
        #pragma parallel block(128)
        for j in range(N):  
            acc = 0
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                acc += a_ik * b[ks,j]
            c[i,j] = acc
    return c


def torch_kernel(a, b):
    return torch.mm(a, b)

def test1():
    for M, K, N in [(4096, 4096, 128), (4096*4, 4096*4, 128)]:
        print(f'M: {M}, N: {N}, K: {K}')
        a_dense = torch.randn(M, K, device='cuda', dtype=torch.float32)
        #a_dense = torch.tril(a_dense)
        a_dense = torch.nn.functional.dropout(a_dense, p=0.9)
        a = a_dense.to_sparse_csr()
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        c_ref = torch_kernel(a, b)

        ms, _, _ = triton.testing.do_bench(lambda: a_dense @ b)
        print(f'dense mm: {ms:.4f} ms')

        for f in [torch_kernel, slap_kernel0]:
            c = f(a, b)
            assert(torch.allclose(c, c_ref, atol=1e-2))
            ms, _, _ = triton.testing.do_bench(lambda: f(a, b))
            print(f'{f.__name__}: {ms:.4f} ms')
