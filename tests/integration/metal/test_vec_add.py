import numpy as np
import numba
import appy
import appy.np_shared as nps
from time import perf_counter

# Numba version: y = x + z
@numba.njit(parallel=True)
def vec_add_numba(x, z, y):
    for i in numba.prange(x.size):
        y[i] = x[i] + z[i]

# APPy version: y = x + z
@appy.jit(verbose_static_rewrite=True, dump_code=True)
def vec_add_appy(x, z, y):
    for i in appy.prange(x.shape[0]):
        y[i] = x[i] + z[i]

# NumPy version: y = x + z
def vec_add_numpy(x, z, y):
    y[:] = x + z

def test_vec_add():
    size = 100_000_000
    x = nps.randn(size, dtype=np.float32)
    z = nps.randn(size, dtype=np.float32)
    y_appy = nps.empty_like(x)
    y_numba = np.empty_like(x)
    y_np = np.empty_like(x)

    # Warmup
    vec_add_appy(x, z, y_appy)
    vec_add_numba(x, z, y_numba)
    vec_add_numpy(x, z, y_np)

    # Check correctness
    assert np.allclose(y_np, y_appy, atol=1e-6)
    assert np.allclose(y_np, y_numba, atol=1e-6)

    # Timing
    t0 = perf_counter()
    vec_add_numpy(x, z, y_np)
    t1 = perf_counter()
    print(f"NumPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    vec_add_appy(x, z, y_appy)
    t1 = perf_counter()
    print(f"APPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    vec_add_numba(x, z, y_numba)
    t1 = perf_counter()
    print(f"Numba: {1000*(t1-t0):.4f} ms")


if __name__ == '__main__':
    test_vec_add()
