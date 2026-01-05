import appy.np_shared as nps
import numpy as np
import appy

@appy.jit(dump_code=True, clear_cache=True)
def store_scalar(out, alpha):
    for i in appy.prange(out.shape[0]):
        out[i] = alpha

def test_store_py_int():
    out = nps.empty((10,), dtype=np.int32)
    store_scalar(out, 10)
    assert np.all(out == 10)

def test_store_py_float():
    out = nps.empty((10,), dtype=np.float32)
    store_scalar(out, 3.3)
    assert np.all(out == 3.3)

def test_store_py_bool():
    out = nps.empty((10,), dtype=np.bool_)
    store_scalar(out, True)
    assert np.all(out == True)

# Error: metalcompute.error: Could not make buffer
def test_store_py_bool1():
    out = nps.empty((10,), dtype=np.bool_)
    store_scalar(out, False)
    assert np.all(out == False)

def test_store_bool():
    out = nps.empty((10,), dtype=np.bool_)
    store_scalar(out, np.bool_(True))
    assert np.all(out == True)

def test_store_int8():
    out = nps.empty((10,), dtype=np.int8)
    store_scalar(out, np.int8(10))
    assert np.all(out == 10)

def test_store_uint8():
    out = nps.empty((10,), dtype=np.uint8)
    store_scalar(out, np.uint8(10))
    assert np.all(out == 10)

def test_store_int16():
    out = nps.empty((10,), dtype=np.int16)
    store_scalar(out, np.int16(10))
    assert np.all(out == 10)

def test_store_uint16():
    out = nps.empty((10,), dtype=np.uint16)
    store_scalar(out, np.uint16(10))
    assert np.all(out == 10)

def test_store_int32():
    out = nps.empty((10,), dtype=np.int32)
    store_scalar(out, np.int32(10))
    assert np.all(out == 10)

def test_store_uint32():
    out = nps.empty((10,), dtype=np.uint32)
    store_scalar(out, np.uint32(10))
    assert np.all(out == 10)

def test_store_float32():
    out = nps.empty((10,), dtype=np.float32)
    store_scalar(out, np.float32(3.3))
    assert np.all(out == 3.3)
