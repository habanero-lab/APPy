import numpy as np
import appy

@appy.jit(backend="cuda", dump_code=True)
def vec_add(a, b, c):
    #pragma parallel for
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]

def test_vec_add():
    N = 1024
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    vec_add(a, b, c)

    assert np.allclose(c, a + b, atol=1e-6), f"Max error: {np.max(np.abs(c - (a + b)))}"
    print("test_vec_add passed")

if __name__ == "__main__":
    test_vec_add()
