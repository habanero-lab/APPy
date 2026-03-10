import numpy as np
import scipy.sparse as sp
import appy
from time import perf_counter

@appy.jit(backend="cuda", dump_code=True)
def spmv_appy(A_data, A_indptr, A_indices, x, y, M):
    #pragma parallel for
    for i in range(M):
        s = 0.0
        for j in range(A_indptr[i], A_indptr[i + 1]):
            s += A_data[j] * x[A_indices[j]]
        y[i] = s

def test_spmv():
    N = 1024
    A = sp.rand(N, N, density=0.05, format='csr', dtype=np.float32)
    x = np.random.rand(N).astype(np.float32)
    y_appy = np.zeros(N, dtype=np.float32)

    spmv_appy(A.data, A.indptr, A.indices, x, y_appy, N)

    y_ref = (A @ x).astype(np.float32)
    assert np.allclose(y_ref, y_appy, atol=1e-4), f"Max error: {np.max(np.abs(y_ref - y_appy))}"

    t0 = perf_counter()
    y_ref = A @ x
    t1 = perf_counter()
    print(f"SciPy: {1000*(t1-t0):.4f} ms")

    y_appy[:] = 0
    t0 = perf_counter()
    spmv_appy(A.data, A.indptr, A.indices, x, y_appy, N)
    t1 = perf_counter()
    print(f"APPy/CUDA: {1000*(t1-t0):.4f} ms")

    print("test_spmv passed")

if __name__ == "__main__":
    test_spmv()
