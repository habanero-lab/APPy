import numpy as np
import appy
import pytest
from time import perf_counter

@appy.jit(backend="cuda", dump_code=True)
def vec_sum_appy(a, s):
    #pragma parallel for
    for i in range(a.shape[0]):
        #pragma atomic
        s[0] += a[i]

@pytest.mark.skip(reason="#pragma atomic not yet implemented in the CUDA backend")
def test_vec_sum():
    N = 1_000_000
    a = np.random.rand(N).astype(np.float32)
    s = np.zeros(1, dtype=np.float32)

    vec_sum_appy(a, s)

    ref = a.sum()
    assert np.allclose(s[0], ref, rtol=1e-3), f"Got {s[0]}, expected {ref}"

    t0 = perf_counter()
    s_np = a.sum()
    t1 = perf_counter()
    print(f"NumPy: {1000*(t1-t0):.4f} ms")

    s[:] = 0
    t0 = perf_counter()
    vec_sum_appy(a, s)
    t1 = perf_counter()
    print(f"APPy/CUDA: {1000*(t1-t0):.4f} ms")

    print("test_vec_sum passed")

if __name__ == "__main__":
    test_vec_sum()
