import numpy as np
import appy
from time import perf_counter

@appy.jit(backend="cuda", dump_code=True)
def mat_add_appy(X, Z, Y):
    #pragma parallel for
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i, j] = X[i, j] + Z[i, j]

def mat_add_numpy(X, Z, Y):
    Y[:] = X + Z

def test_mat_add():
    rows, cols = 1024, 1024
    X = np.random.rand(rows, cols).astype(np.float32)
    Z = np.random.rand(rows, cols).astype(np.float32)
    Y_appy = np.zeros((rows, cols), dtype=np.float32)
    Y_np = np.zeros((rows, cols), dtype=np.float32)

    # Warmup
    mat_add_appy(X, Z, Y_appy)
    mat_add_numpy(X, Z, Y_np)

    assert np.allclose(Y_np, Y_appy, atol=1e-5), f"Max error: {np.max(np.abs(Y_np - Y_appy))}"

    t0 = perf_counter()
    mat_add_numpy(X, Z, Y_np)
    t1 = perf_counter()
    print(f"NumPy: {1000*(t1-t0):.4f} ms")

    t0 = perf_counter()
    mat_add_appy(X, Z, Y_appy)
    t1 = perf_counter()
    print(f"APPy/CUDA: {1000*(t1-t0):.4f} ms")

    print("test_mat_add passed")

if __name__ == "__main__":
    test_mat_add()
