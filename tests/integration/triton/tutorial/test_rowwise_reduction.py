import numpy as np
import appy

@appy.jit(backend="triton", dump_code=True)
def matrix_rowwise_sum(a, b):
    #pragma parallel for
    for i in range(a.shape[0]):
        #pragma simd
        s = 0.0
        for j in range(a.shape[1]):
            s += a[i, j]
        b[i] = s

def test_rowwise_sum():
    a = np.random.rand(100, 100)
    b = np.empty(100)
    matrix_rowwise_sum(a, b)
    assert np.allclose(b, a.sum(axis=1))

@appy.jit(backend="triton", dump_code=True)
def matrix_rowwise_max(a, b):
    #pragma parallel for
    for i in range(a.shape[0]):
        
        s = -1e308
        #pragma simd
        for j in range(a.shape[1]):
            s = max(s, a[i, j])
        b[i] = s

def test_rowwise_max():
    a = np.random.rand(100, 100)
    b = np.empty(100)
    matrix_rowwise_max(a, b)
    assert np.allclose(b, a.max(axis=1))


@appy.jit(backend="triton", dump_code=True)
def matrix_rowwise_min(a, b):
    #pragma parallel for
    for i in range(a.shape[0]):
        
        s = 1e308
        #pragma simd
        for j in range(a.shape[1]):
            s = min(s, a[i, j])
        b[i] = s

def test_rowwise_min():
    a = np.random.rand(100, 100)
    b = np.empty(100)
    matrix_rowwise_min(a, b)
    assert np.allclose(b, a.min(axis=1))