import torch
import appy
import triton
import triton.language as tl

const = tl.constexpr

#@appy.jit(tune=['BLOCK'])
def kernel(a_rowptrs, a_cols, a_vals, b_crows, b_cols, b_vals, c, BLOCK):
    for i in range(c.shape[0]):  #pragma parallel 
        acc = torch.zeros(c.shape[1], device=c.device, dtype=c.dtype)
        for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
            k = a_cols[ki]
            a_ik = a_vals[ki]
            for ji in range(b_crows[i], b_crows[i+1], BLOCK):
                j_idx = range(ji, min(ji+BLOCK, b_crows[i+1]))
                j = b_cols[j_idx]
                b_kj = b_vals[j_idx]
                acc[j] += a_ik * b_kj
        c[i,:] = acc
                

def _triton_kernel(a_rowptrs, a_cols, a_vals, b_crows, b_cols, b_vals, c, 
                   BLOCK: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr)
    i = tl.program_id(0)
    acc = tl.zeros()


for M in [1024]:
    K = M
    N = 32
    print(f'M: {M}, N: {N}, K: {K}')
    a_dense = torch.randn(M, K, device='cuda', dtype=torch.float32)
    a_dense = torch.tril(a_dense)
    a = a_dense.to_sparse_csr()
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        c = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        f(a.crow_indices(), a.col_indices(), a.values(), b, c, 64)
        # print(c)
        # print(a @ b)
        assert(torch.allclose(c, torch.mm(a, b), atol=1e-3))
