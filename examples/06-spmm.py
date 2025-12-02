import torch
import appy

@appy.jit(dump_code=True)
def kernel_appy(A_data, A_indptr, A_indices, B, y, M): 
    #pragma parallel for
    for i in range(M):
        start, end = A_indptr[i], A_indptr[i+1]
        y[i, :] = 0
        for k in range(start, end):
            y[i, :] += A_data[k] * B[A_indices[k], :]


def kernel_appy_torch(A, x):
    y = torch.empty([A.shape[0], x.shape[1]], dtype=A.dtype, device=A.device)
    """Call APPy kernel using CSR tensor parts."""
    kernel_appy(
        A.values(),       # CSR data
        A.crow_indices(), # CSR row pointer (indptr)
        A.col_indices(),  # CSR column indices
        x,
        y,
        A.shape[0],
    )
    return y


def kernel_lib(A, B):
    return A @ B


def test():
    N = 10000
    for sparsity in [0.0001, 0.001, 0.01, 0.1]:
        # Random sparse matrix in CSR format
        A = torch.randn(N, N, device='cuda', dtype=torch.float64)
        mask = torch.rand(N, N, device='cuda') < sparsity
        A = (A * mask).to_sparse_csr()
        B = torch.randn(N, 200, device='cuda', dtype=torch.float64)

        print(f"A shape: {A.shape}, A nnz: {A._nnz()}, dtype: {A.dtype}")

        # Reference using PyTorch's built-in sparse CSR matvec
        y_ref = A @ B
        # APPy kernel call
        y = kernel_appy_torch(A, B)

        torch.testing.assert_close(y, y_ref, rtol=1e-6, atol=1e-6)

        # Bench

        import time
        t0 = time.time(); _ = A @ B; torch.cuda.synchronize(); ms0 = (time.time()-t0)*1000
        t1 = time.time(); _ = kernel_appy_torch(A, B); torch.cuda.synchronize(); ms1 = (time.time()-t1)*1000
        print(f"PyTorch CSR matvec: {ms0:.4f} ms | APPy kernel: {ms1:.4f} ms")


if __name__ == '__main__':
    test()