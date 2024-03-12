import appy
import cupy
import cupyx
import torch


@appy.jit(auto_simd=True)
def kernel_appy0(A_data, A_indptr, A_indices, B, N, K):
    y = appy.empty([N, K], dtype=B.dtype)
    #pragma parallel for
    for i in range(N):
        start, end = A_indptr[i], A_indptr[i+1]
        #pragma :K=>le(128)
        y[i, :K] = 0
        for j in range(start, end):
            #pragma :K=>le(128)
            y[i, :K] += A_data[j] * B[A_indices[j], :K]
    return y


@appy.jit(auto_simd=True)
def kernel_appy(A_data, A_indptr, A_indices, B, N, K):
    y = appy.empty([N, K], dtype=B.dtype)
    #pragma parallel for
    for i in range(N):
        start, end = A_indptr[i], A_indptr[i+1]
        #pragma :K=>le(128)
        y[i, :K] = 0.0
        #pragma :K=>le(128)
        acc = y[i, :K]
        for j in range(start, end):
            #pragma :K=>le(128)
            acc += A_data[j] * B[A_indices[j], :K]
        #pragma :K=>le(128)
        y[i, :K] = acc
    return y


def kernel_lib(A, B):
    return A @ B


def test():
    N = 20000
    K = 100
    for sparsity in [0.0001, 0.0004, 0.001, 0.004, 0.01, 0.04, 0.1]:    
        A = cupyx.scipy.sparse.random(N, N, sparsity).tocsr()
        B = cupy.random.randn(N, K)
        print(f'A shape: {A.shape}, sparsity: {sparsity} A nnz: {A.count_nonzero()}')
        
        A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, size=(N, N), device='cuda')
        B = torch.as_tensor(B, device='cuda')
        
        y_ref = kernel_lib(A, B)
        y = kernel_appy0(A.values(), A.crow_indices(), A.col_indices(), B, N, K)
        assert appy.utils.allclose(y, y_ref, atol=1e-6)
        ms0 = appy.utils.bench(lambda: kernel_lib(A, B))
        ms1 = appy.utils.bench(lambda: kernel_appy0(A.values(), A.crow_indices(), A.col_indices(), B, N, K))
        print(f"kernel_lib: {ms0:.4f} ms")
        print(f"kernel_appy: {ms1:.4f} ms")
        print(f'speedup over lib: {(ms0/ms1):.3f}')

        A = A.to_dense()
        ms2 = appy.utils.bench(lambda: kernel_lib(A, B))
        print(f"kernel_lib (dense): {ms2:.4f} ms")
        # A, B = A.cpu(), B.cpu()
        # ms2 = appy.utils.bench(lambda: kernel_lib(A, B))
        # print(f"kernel_lib (CPU): {ms2:.4f} ms")


if __name__ == "__main__":
    test()
