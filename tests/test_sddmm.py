import torch
import slap
import triton
import triton.language as tl

#@slap.jit(tune=['BLOCK'])
def kernel(a_rowptrs, a_cols, a_vals, b, c, d_vals, BLOCK):
    for i in range(b.shape[0]):  #pragma parallel 
        for ji in range(a_rowptrs[i], a_rowptrs[i+1], BLOCK):
            idx = range(ji, min(ji+BLOCK, a_rowptrs[i+1]))
            js = a_cols[idx]
            d_acc = torch.zeros(len(idx), device=b.device, dtype=b.dtype)
            a_ij = a_vals[idx]
            
            for k in range(b.shape[1]):
                b_ik = b[i,k]
                c_kj = c[k,js]
                d_ij = a_ij * b_ik * c_kj
                d_acc += d_ij
            d_vals[idx] = d_acc

@triton.jit        
def _triton_kernel(a_rowptrs, a_cols, a_vals, b, c, d_vals, 
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BLOCK: tl.constexpr, e, f):
    i = tl.program_id(0)
    row_start = tl.load(a_rowptrs+i)
    row_end = tl.load(a_rowptrs+i+1)
    for ji in range(row_start, row_end, BLOCK):
        idx = ji + tl.arange(0, BLOCK)
        mask = idx < row_end
        js = tl.load(a_cols+idx)
        d_acc = tl.zeros([BLOCK], dtype=tl.float32)
        
        offsets = tl.load(e)
        tl.store(d_acc+offsets, js)

        a_ij = tl.load(a_vals+idx, mask=mask, other=0)
        for k in range(K):
            b_ik = tl.load(b+i*K+k)
            c_kj = tl.load(c+k*N+js, mask=mask, other=0)
            d_ij = a_ij * b_ik * c_kj
            d_acc += d_ij
            
        tl.store(d_vals+idx, d_acc, mask=mask)


def triton_kernel(a_rowptrs, a_cols, a_vals, b, c, d_vals, BLOCK):
    M, K = b.shape
    K, N = c.shape
    grid = (M,)
    e = torch.arange(BLOCK, device=a.device)
    f = torch.zeros(BLOCK, device=a.device, dtype=a.dtype)
    _triton_kernel[grid](a_rowptrs, a_cols, a_vals, b, c, d_vals, M, N, K, BLOCK, e, f)
        

for M in [10240*2]:
    N = M
    K = 64
    print(f'M: {M}, N: {N}, K: {K}')
    a_dense = torch.randn(M, N, device='cuda', dtype=torch.float32)
    a_dense = torch.nn.functional.dropout(a_dense, p=0.9)
    #a_dense = torch.tril(a_dense)
    a = a_dense.to_sparse_csr()
    a_coo = a.to_sparse_coo()
    b = torch.randn(M, K, device='cuda', dtype=torch.float32)
    c = torch.randn(K, N, device='cuda', dtype=torch.float32)

    ms, _, _ = triton.testing.do_bench(lambda: (a * (b @ c)))
    print(f'torch: {ms} ms')
    ms, _, _ = triton.testing.do_bench(lambda: b @ c)
    print(f'torch mm: {ms} ms')


    for f in [triton_kernel]:
        d_vals = a.values().clone()
        BLOCK = 128
        print('call before')
        f(a.crow_indices(), a.col_indices(), a.values(), b, c, d_vals, BLOCK)
        print('call after')
        #d_vals_ref = torch.sparse.sampled_addmm(a, b, c).values()  # this is not really SDDMM, it uses spy(a)
        d_vals_ref = (a * (b @ c)).values()
        print(d_vals)
        print(d_vals_ref)
        assert(torch.allclose(d_vals, d_vals_ref, atol=1e-3))
        ms, _, _ = triton.testing.do_bench(lambda: f(a.crow_indices(), a.col_indices(), a.values(), b, c, d_vals, BLOCK))
        print(f'kernel: {ms} ms')
