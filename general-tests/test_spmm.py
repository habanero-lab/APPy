import torch
import triton
import slap
from torch import arange, zeros, empty, sum

def slap_kernel0(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty([M, N], device='cuda', dtype=a.dtype)
    acc_dtype = a.dtype
    if a.dtype == torch.float16:
        acc_dtype = torch.float32
    _slap_kernel0(a.crow_indices(), a.col_indices(), a.values(), b, c, M, K, N, N, acc_dtype)
    return c

@slap.jit
def _slap_kernel0(a_rowptrs, a_cols, a_vals, b, c, M, K, N, BN, acc_dtype):
    for i in range(M):  #pragma parallel 
        for j in range(0, N, BN):  #pragma parallel 
            acc = zeros(BN, device='cuda', dtype=acc_dtype)
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                acc += a_ik * b[ks,j:j+BN]
            c[i,j:j+BN] = acc


def slap_kernel(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty([M, N], device='cuda', dtype=a.dtype)
    _slap_kernel(a.crow_indices(), a.col_indices(), a.values(), b, c, M, K, N)
    return c

#@slap.jit
def _slap_kernel(a_rowptrs, a_cols, a_vals, b, c, M, K, N, BN=128):
    for i in range(M):  #pragma parallel 
        for j in range(N):  #pragma parallel block(BN)
            acc = 0
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                acc += a_ik * b[ks,j]
            c[i,j] = acc

#@slap.jit
def mykernel_ops(a_rowptrs, a_cols, a_vals, b, c, M, K, N, BN=128):
    for i in range(M):  #pragma parallel     
        c[i,:N] = 0.0
        for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
            a_ik = a_vals[ki]
            ks = a_cols[ki]
            #pragma par_dim(:N:BN)    # this is not supported yet, there cannot be statements between parallel pragmas
            c[i,:N] += a_ik * b[ks,:N]

def torch_kernel(a, b):
    return torch.mm(a, b)

def test1():
    for dtype in [torch.float16, torch.float32]:
        for M, K, N in [(4096, 4096, 128), (4096*4, 4096*4, 128)]:
            print(f'M: {M}, N: {N}, K: {K}')
            a_dense = torch.randn(M, K, device='cuda', dtype=dtype)
            #a_dense = torch.tril(a_dense)
            a_dense = torch.nn.functional.dropout(a_dense, p=0.9)
            a = a_dense.to_sparse_csr()
            b = torch.randn(K, N, device='cuda', dtype=dtype)
            c_ref = torch_kernel(a, b)

            ms, _, _ = triton.testing.do_bench(lambda: a_dense @ b)
            print(f'dense mm: {ms:.4f} ms')

            for f in [torch_kernel, slap_kernel0]:
                c = f(a, b)
                if dtype == torch.float16:
                    assert(torch.allclose(c, c_ref, atol=5, rtol=0.1))
                else:
                    assert(torch.allclose(c, c_ref, atol=0.5, rtol=0.01))
                ms, _, _ = triton.testing.do_bench(lambda: f(a, b))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()