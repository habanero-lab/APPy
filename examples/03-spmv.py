import torch
import appy


@appy.jit(dump_code=True)
def kernel_appy(A_data, A_indptr, A_indices, x, M, N):
    y = torch.empty(M, dtype=x.dtype, device='cuda')
    #pragma parallel for
    for i in range(M):
        start, end = A_indptr[i], A_indptr[i+1]
        s = 0.0
        #pragma simd
        for j in range(start, end):
            s += A_data[j] * x[A_indices[j]]
        y[i] = s
    return y


def kernel_appy_torch(A, x):
    """Call APPy kernel using CSR tensor parts."""
    return kernel_appy(
        A.values(),       # CSR data
        A.crow_indices(), # CSR row pointer (indptr)
        A.col_indices(),  # CSR column indices
        x,
        A.shape[0],
        A.shape[1],
    )


def test_torch_with_appy():
    N = 10000
    for sparsity in [0.0001, 0.001, 0.01, 0.1]:
        # Random sparse matrix in CSR format
        dense = torch.randn(N, N, device='cuda', dtype=torch.float64)
        mask = torch.rand(N, N, device='cuda') < sparsity
        dense = dense * mask
        A = dense.to_sparse_csr()
        x = torch.randn(N, device='cuda', dtype=torch.float64)

        print(f"A shape: {A.shape}, A nnz: {A._nnz()}, dtype: {A.dtype}")

        # Reference using PyTorch's built-in sparse CSR matvec
        y_ref = A @ x
        # APPy kernel call
        y = kernel_appy_torch(A, x)

        torch.testing.assert_close(y, y_ref, rtol=1e-6, atol=1e-6)

        # Bench

        import time
        t0 = time.time(); _ = A @ x; torch.cuda.synchronize(); ms0 = (time.time()-t0)*1000
        t1 = time.time(); _ = kernel_appy_torch(A, x); torch.cuda.synchronize(); ms1 = (time.time()-t1)*1000
        print(f"PyTorch CSR matvec: {ms0:.4f} ms | APPy kernel: {ms1:.4f} ms")


if __name__ == "__main__":
    test_torch_with_appy()
