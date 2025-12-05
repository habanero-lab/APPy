import numpy as np
import numba
import appy
import appy.np_shared as nps
from time import perf_counter

@numba.njit(parallel=True)
def gelu_numba(x):
    y = np.empty_like(x)
    for i in numba.prange(x.size):
        xi = x[i]
        x3 = xi * xi * xi
        t = np.tanh(0.79788456 * (xi + 0.044715 * x3))
        y[i] = 0.5 * xi * (1 + t)

    return y


@appy.jit(verbose_static_rewrite=True)
def gelu_appy(x, y):
    y = nps.empty_like(x)
    for i in appy.prange(x.size):
        xi = x[i]
        x3 = xi * xi * xi
        t = np.tanh(0.79788456 * (xi + 0.044715 * x3))
        y[i] = 0.5 * xi * (1 + t)

    return y


def gelu_numpy(x):
    x3 = x * x * x
    t = np.tanh(0.79788456 * (x + 0.044715 * x3))
    y = 0.5 * x * (1 + t)
    return y


def test():
    x = nps.randn(20_000_000, dtype=np.float32)
    y = nps.randn(20_000_000, dtype=np.float32)

    y_np = gelu_numpy(x)
    t0 = perf_counter()
    y_appy = gelu_appy(x, y)
    t1 = perf_counter()
    print(f"APPy (with JIT): {1000*(t1-t0):.4f} ms")
    y_numba = gelu_numba(x.arr)

    assert np.allclose(y_np, y_appy, atol=1e-6)
    assert np.allclose(y_np, y_numba, atol=1e-6)

    # Timing
    t0 = perf_counter()
    _ = gelu_numpy(x)
    t1 = perf_counter()
    print(f"NumPy: {1000*(t1-t0):.4f} ms")
    t0 = perf_counter()
    _ = gelu_appy(x, y)
    t1 = perf_counter()
    print(f"APPy: {1000*(t1-t0):.4f} ms")
    t0 = perf_counter()
    _ = gelu_numba(x.arr)
    t1 = perf_counter()
    print(f"Numba: {1000*(t1-t0):.4f} ms")


if __name__ == '__main__':
    test()