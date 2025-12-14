import numpy as np
import numba
import appy
import appy.np_shared as nps
from time import perf_counter

# Numba version: Y = X + Z
@numba.njit(parallel=True)
def mat_add_numba(X, Z, Y):
    for i in numba.prange(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i, j] = X[i, j] + Z[i, j]


# APPy version with flattened 1D indexing
@appy.jit(verbose_static_rewrite=True, dump_code=True)
def mat_add_appy1(X, Z, Y):
    n_rows, n_cols = X.shape
    size = n_rows * n_cols
    for idx in appy.prange(size):
        i = idx // n_cols
        j = idx % n_cols
        Y[i, j] = X[i, j] + Z[i, j]


@appy.jit(verbose_static_rewrite=True, dump_code=True)
def mat_add_appy(X, Z, Y):
    for i in appy.prange(X.shape[0]):
        for j in appy.prange(X.shape[1]):
            Y[i, j] = X[i, j] + Z[i, j]

# NumPy version: Y = X + Z
def mat_add_numpy(X, Z, Y):
    Y[:] = X + Z

def test_mat_add():
    rows, cols = 4096, 4096  # Large enough to test GPU/parallel performance
    X = nps.randn(rows, cols, dtype=np.float32)
    Z = nps.randn(rows, cols, dtype=np.float32)
    Y_appy = nps.empty_like(X)
    Y_numba = np.empty_like(X)
    Y_np = np.empty_like(X)

    # Warmup
    mat_add_appy(X, Z, Y_appy)
    mat_add_numba(X, Z, Y_numba)
    mat_add_numpy(X, Z, Y_np)

    # Check correctness
    assert np.allclose(Y_np, Y_appy, atol=1e-6)
    assert np.allclose(Y_np, Y_numba, atol=1e-6)

    # Timing
    t0 = perf_counter()
    mat_add_numpy(X, Z, Y_np)
    t1 = perf_counter()
    print(f"NumPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    mat_add_appy(X, Z, Y_appy)
    t1 = perf_counter()
    print(f"APPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    mat_add_numba(X, Z, Y_numba)
    t1 = perf_counter()
    print(f"Numba: {1000*(t1-t0):.4f} ms")


if __name__ == '__main__':
    test_mat_add()
