import numpy as np
import numba
import appy
import appy.np_shared as nps
from time import perf_counter

# Numba version: y = a * x + y
@numba.njit(parallel=True)
def saxpy_numba(a, x, y):
    for i in numba.prange(x.size):
        y[i] = a * x[i] + y[i]

# APPy version: y = a * x + y
@appy.jit(verbose_static_rewrite=True, dump_code=True)
def saxpy_appy(a, x, y):
    for i in appy.prange(x.shape[0]):
        y[i] = a * x[i] + y[i]

# NumPy version: y = a * x + y
def saxpy_numpy(a, x, y):
    y[:] = a * x + y

def test_saxpy():
    size = 20_000_007
    a = np.float32(2.5)
    x = nps.randn(size, dtype=np.float32)
    y_appy = nps.randn(size, dtype=np.float32)
    y_numba = y_appy.copy()
    y_np = y_appy.copy()

    # Warmup
    saxpy_appy(a, x, y_appy)
    saxpy_numba(a, x, y_numba)
    saxpy_numpy(a, x, y_np)

    # Check correctness
    assert np.allclose(y_np, y_appy, atol=1e-6)
    assert np.allclose(y_np, y_numba, atol=1e-6)

    # Timing
    t0 = perf_counter()
    saxpy_numpy(a, x, y_np)
    t1 = perf_counter()
    print(f"NumPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    saxpy_appy(a, x, y_appy)
    t1 = perf_counter()
    print(f"APPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    saxpy_numba(a, x, y_numba)
    t1 = perf_counter()
    print(f"Numba: {1000*(t1-t0):.4f} ms")

if __name__ == '__main__':
    test_saxpy()
