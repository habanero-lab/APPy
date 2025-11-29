import numpy as np
import appy

@appy.jit(backend="triton", dump_code=True, dry_run=0)
def vec_sum(a):
    sum = 0.0
    #pragma parallel for simd shared(sum)
    for i in range(a.shape[0]):
        sum += a[i]
    return sum

def test_sum():
    n = 1000
    a = np.random.rand(n)
    b = vec_sum(a)
    assert np.allclose(b, a.sum())

@appy.jit(backend="triton", dump_code=True, dry_run=0)
def vec_max(a):
    m = -np.inf
    #pragma parallel for simd shared(m)
    for i in range(a.shape[0]):
        m = max(m, a[i])
    return m

def test_max():
    n = 1000
    a = np.random.rand(n)
    b = vec_max(a)
    assert np.allclose(b, a.max())