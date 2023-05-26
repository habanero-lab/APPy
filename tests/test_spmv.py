import torch
import bmp
import triton
import triton.language as tl

print(torch.__version__)
print(triton.__version__)
print(triton.__file__)

@triton.jit
def _triton_kernel(a_rowptrs, a_cols, a_vals, b, c, M: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    row_start = tl.load(a_rowptrs+i)
    row_end = tl.load(a_rowptrs+i+1)
    acc = 0.0
    for ji in range(row_start, row_end, BLOCK):
        idx = ji + tl.arange(0, BLOCK) 
        mask = idx < row_end
        a_ij = tl.load(a_vals+idx, mask=mask, other=0)
        j = tl.load(a_cols+idx, mask=mask, other=0)
        b_j = tl.load(b+j, mask=mask, other=0)  # gather
        acc += tl.sum(a_ij * b_j, axis=0)
    tl.store(c+i, acc)

def triton_kernel(a_rowptrs, a_cols, a_vals, b, c, M, BLOCK):
    grid = (M,)
    _triton_kernel[grid](a_rowptrs, a_cols, a_vals, b, c, M, BLOCK)


#@bmp.jit(tune=['BLOCK'])
def kernel(a_rowptrs, a_cols, a_vals, b, c, M, BLOCK):
    for i in range(M):  #pragma parallel
        row_start = a_rowptrs[i]
        row_end = a_rowptrs[i+1]
        acc = 0.0
        for ji in range(row_start, row_end, BLOCK):
            idx = range(ji, min(ji+BLOCK, row_end))
            a_ij = a_vals[idx]
            j = a_cols[idx]
            b_j = b[j]  # gather
            acc += torch.sum(a_ij * b_j)
        c[i] = acc


for M in [1024*20]:
    N = M
    print(f'M: {M}, N: {N}')
    a_dense = torch.randn(M, N, device='cuda', dtype=torch.float32)
    a_dense = torch.tril(a_dense)
    a = a_dense.to_sparse_csr()
    b = torch.randn(N, device='cuda', dtype=torch.float32)

    ms, _, _ = triton.testing.do_bench(lambda: torch.mv(a, b))
    print(f'torch: {ms} ms')

    for f in [triton_kernel, kernel]:
        c = torch.zeros(M, device='cuda', dtype=torch.float32)
        BLOCK = 128
        f(a.crow_indices(), a.col_indices(), a.values(), b, c, M, BLOCK)
        # print(c)
        # print(a @ b)
        assert(torch.allclose(c, torch.mv(a_dense, b), atol=1e-3))
        ms, _, _ = triton.testing.do_bench(lambda: f(a.crow_indices(), a.col_indices(), a.values(), b, c, M, BLOCK))
        print(f'kernel: {ms} ms')
