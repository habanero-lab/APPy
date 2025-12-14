import appy.np_shared as nps
import numpy as np
import appy

@appy.jit
def store_int(out, alpha):
    for i in range(out.shape[0]):
        out[i] = alpha

def test_store_int():
    out = nps.empty((10,), dtype=np.int32)
    store_int(out, 10)
    assert np.all(out == 10)

@appy.jit
def store_float(out, alpha):
    for i in range(out.shape[0]):
        out[i] = alpha

def test_store_float():
    out = nps.empty((10,), dtype=np.float32)
    store_float(out, 3.3)
    assert np.all(out == 3.3)