import torch
import slap

#@slap.jit(tune=['BLOCK'])
def kernel(a_rowptrs, a_cols, a_vals, b, c, BLOCK):
    for i in range(a.shape[0]):  #pragma parallel 
        for j in range(0, b.shape[1], BLOCK):  #pragma parallel 
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                c[i,j:j+BLOCK] += a_ik * b[ks,j:j+BLOCK]


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
