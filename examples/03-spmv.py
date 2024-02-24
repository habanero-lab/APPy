import appy
import cupy
import cupyx
appy.tensorlib = cupy


@appy.jit
def kernel_appy(A_data, A_indptr, A_indices, x, M, N):
    y = appy.empty(M, dtype=x.dtype)
    #pragma parallel for
    for i in range(M):
        start, end = A_indptr[i], A_indptr[i+1]
        y[i] = 0.0
        #pragma simd
        for j in range(start, end):
            #col = A_indices[j]
            #y[i] += A_data[j] * x[col]
            y[i] += A_data[j] * x[A_indices[j]]
    return y


def kernel_lib(A, x):
    return A @ x


def test():
    N = 10000
    for sparsity in [0.0001, 0.001, 0.01, 0.1]:    
        A = cupyx.scipy.sparse.random(N, N, sparsity).tocsr()
        x = cupy.random.randn(N)
        print(f'A shape: {A.shape}, A nnz: {A.count_nonzero()}')
        y_ref = kernel_lib(A, x)
        y = kernel_appy(A.data, A.indptr, A.indices, x, N, N)
        assert appy.utils.allclose(y, y_ref, atol=1e-6)
        ms0 = appy.utils.bench(lambda: kernel_lib(A, x))
        ms1 = appy.utils.bench(lambda: kernel_appy(A.data, A.indptr, A.indices, x, N, N))
        print(f"kernel_lib: {ms0:.4f} ms")
        print(f"kernel_appy: {ms1:.4f} ms")


if __name__ == "__main__":
    test()
