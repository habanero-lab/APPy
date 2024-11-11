import appy
import cupyx
import torch


@appy.jit(auto_simd=True)
def kernel_appy(A_data, A_indptr, A_indices, B, M, N):
    y = appy.empty([M, N], dtype=B.dtype)
    #pragma parallel for
    for i in range(M):
        start, end = A_indptr[i], A_indptr[i+1]
        y[i, :N] = 0
        for k in range(start, end):
            y[i, :N] += A_data[k] * B[A_indices[k], :N]
    return y


def kernel_lib(A, B):
    return A @ B


def test():
    M = K = 20000
    N = 256
    print(f'Computing SpMM: A ({M}x{K}, sparse) @ B ({K}x{N}, dense)')
    #for sparsity in [0.0001, 0.0004, 0.001, 0.004, 0.01, 0.04, 0.1]:    
    for sparsity in [0.0001, 0.0004, 0.001, 0.01]:    
        A = cupyx.scipy.sparse.random(M, K, sparsity).tocsr()
        print(f'A sparsity: {sparsity}, nnz: {A.count_nonzero()}')
        #print(f'B shape: {B.shape}')
        A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, size=(M, K), device='cuda')
        B = torch.randn(K, N, dtype=A.dtype, device='cuda')
        
        y_ref = kernel_lib(A, B)
        y = kernel_appy(A.values(), A.crow_indices(), A.col_indices(), B, M, N)
        assert appy.utils.allclose(y, y_ref, atol=1e-6)
        ms0 = appy.utils.bench(lambda: kernel_lib(A, B))
        ms1 = appy.utils.bench(lambda: kernel_appy(A.values(), A.crow_indices(), A.col_indices(), B, M, N))
        print(f"kernel_lib: {ms0:.2f} ms")
        print(f"kernel_appy: {ms1:.2f} ms")
        print(f'speedup over lib: {(ms0/ms1):.2f}')

        A = A.to_dense()
        ms2 = appy.utils.bench(lambda: kernel_lib(A, B))
        print(f"kernel_lib (dense): {ms2:.2f} ms\n")
        # A, B = A.cpu(), B.cpu()
        # ms2 = appy.utils.bench(lambda: kernel_lib(A, B))
        # print(f"kernel_lib (CPU): {ms2:.4f} ms")


if __name__ == "__main__":
    test()
