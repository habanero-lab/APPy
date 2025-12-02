'''
This test attempts to test invalid pragma syntax, which should fail.
'''
import numpy as np
import appy


@appy.jit(backend="triton", dump_code=True)
def kernel1(a, b):
    c = np.empty_like(a)
    #pragma parallel for simd my_directive
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c


@appy.jit(backend="triton", dump_code=True)
def kernel2(a, b):
    c = np.empty_like(a)
    #pragma parallel for simd 

    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c


import pytest

def test():
    a = np.ones(10)
    b = np.ones(10)

    # kernel2 should succeed
    assert np.allclose(kernel2(a, b), a + b)

    # kernel1 must raise an exception
    with pytest.raises(Exception) as excinfo:
        kernel1(a, b)

    print(excinfo.value)
    # check that "my_directive" appears in the error message
    assert "my_directive" in str(excinfo.value)

    
