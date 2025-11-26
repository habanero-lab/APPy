import numpy as np
import appy

@appy.jit(backend="triton", dump_code=True)
def vec_sum(a):
    sum = 0.0
    #pragma parallel for simd shared(sum)
    for i in range(a.shape[0]):
        sum += a[i]
    return sum

def test():
    n = 1000
    a = np.random.rand(n)
    b = vec_sum(a)
    assert np.allclose(b, a.sum())