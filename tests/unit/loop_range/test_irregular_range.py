import numpy as np
import appy
import pytest

@appy.jit(backend="triton", dump_code=True)
def kernel_appy(a, b):
    c = np.empty_like(a)
    #pragma parallel for simd
    for i in range(0, a.shape[0], 2):
        c[i] = a[i] + b[i]
    return c

def test():
    a = np.ones(100)
    b = np.ones(100)
    
    with pytest.raises(Exception) as excinfo:
        kernel_appy(a, b)

    print(excinfo.value)
    assert "range" in str(excinfo.value) and "step" in str(excinfo.value)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
