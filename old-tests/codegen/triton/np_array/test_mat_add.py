import numpy as np
import appy


@appy.jit(backend="triton", dump_code=True)
def kernel_appy(a, b):
    c = np.empty_like(a)
    #pragma parallel for
    for i in range(a.shape[0]):
        #pragma simd
        for j in range(a.shape[1]):
            c[i, j] = a[i, j] + b[i, j]
    return c


def test_mat_add():
    a = np.ones((100, 100))
    b = np.ones((100, 100))
    c = kernel_appy(a, b)
    assert np.allclose(c, a + b)

