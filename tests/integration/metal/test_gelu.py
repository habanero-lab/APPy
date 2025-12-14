import numpy as np
import numba
import appy
import appy.np_shared as nps
from time import perf_counter

# Numba version: write into preallocated y
@numba.njit(parallel=True)
def gelu_numba(x, y):
    for i in numba.prange(x.size):
        xi = x[i]
        x3 = xi * xi * xi
        t = np.tanh(0.79788456 * (xi + 0.044715 * x3))
        y[i] = 0.5 * xi * (1 + t)

# APPy version: write into preallocated y
@appy.jit(verbose_static_rewrite=True, dump_code=True)
def gelu_appy(x, y):
    for i in appy.prange(x.shape[0]):
        xi = x[i]
        x3 = xi * xi * xi
        t = np.tanh(0.79788456 * (xi + 0.044715 * x3))
        y[i] = 0.5 * xi * (1 + t)

# NumPy version: write into preallocated y
def gelu_numpy(x, y):
    x3 = x * x * x
    t = np.tanh(0.79788456 * (x + 0.044715 * x3))
    y[:] = 0.5 * x * (1 + t)

def test():
    size = 20_000_000
    x = nps.randn(size, dtype=np.float32)
    y_appy = nps.empty_like(x)
    y_numba = np.empty_like(x)
    y_np = np.empty_like(x)

    # Warmup
    gelu_appy(x, y_appy)
    gelu_numba(x, y_numba)
    gelu_numpy(x, y_np)

    # Check correctness
    assert np.allclose(y_np, y_appy, atol=1e-6)
    assert np.allclose(y_np, y_numba, atol=1e-6)

    # Timing
    t0 = perf_counter()
    gelu_numpy(x, y_np)
    t1 = perf_counter()
    print(f"NumPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    gelu_appy(x, y_appy)
    t1 = perf_counter()
    print(f"APPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    gelu_numba(x, y_numba)
    t1 = perf_counter()
    print(f"Numba: {1000*(t1-t0):.4f} ms")

if __name__ == '__main__':
    test()
