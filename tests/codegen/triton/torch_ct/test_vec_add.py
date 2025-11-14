import os
import ast
from pathlib import Path
import torch as np
import appy

@appy.jit(backend="triton", dump_code=True)
def kernel_appy(a, b):
    c = np.empty_like(a)
    for i in appy.prange(a.shape[0], simd=True):
        c[i] = a[i] + b[i]
    return c

def test_vec_add():
    """Verify that APPy generates the expected Triton code for a simple add kernel."""
    # Define the kernel under test

    a = np.ones(100)
    b = np.ones(100)
    c = kernel_appy(a, b)
    assert np.allclose(c, a + b)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
