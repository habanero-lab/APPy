import numpy as np
import appy
from time import perf_counter

@appy.jit(backend="cuda", dump_code=True)
def saxpy_appy(a, x, y):
    #pragma parallel for
    for i in range(x.shape[0]):
        y[i] = a * x[i] + y[i]

def saxpy_numpy(a, x, y):
    y[:] = a * x + y

def test_saxpy():
    size = 1_000_000
    a = np.float32(2.5)
    x = np.random.rand(size).astype(np.float32)
    y_appy = np.random.rand(size).astype(np.float32)
    y_np = y_appy.copy()

    # Warmup
    saxpy_appy(a, x, y_appy)
    saxpy_numpy(a, x, y_np)

    assert np.allclose(y_np, y_appy, atol=1e-5), f"Max error: {np.max(np.abs(y_np - y_appy))}"

    t0 = perf_counter()
    saxpy_numpy(a, x, y_np)
    t1 = perf_counter()
    print(f"NumPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    saxpy_appy(a, x, y_appy)
    t1 = perf_counter()
    print(f"APPy/CUDA: {1000*(t1-t0):.4f} ms")

    print("test_saxpy passed")

if __name__ == "__main__":
    test_saxpy()
