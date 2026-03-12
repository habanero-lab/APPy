import numpy as np
import metalcompute as mc

device = mc.Device()

array_to_buffer = {}

# Create a global generator instance
_rng = np.random.default_rng()

def empty(shape, dtype=np.float64):
    dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    buf = device.buffer(n * dtype.itemsize)
    arr = np.frombuffer(buf, dtype=dtype, count=n).reshape(shape)
    array_to_buffer[arr.ctypes.data] = buf
    return arr

def empty_like(x, dtype=None):
    return empty(x.shape, dtype or x.dtype)

def zeros(shape, dtype=np.float64):
    out = empty(shape, dtype=dtype)
    return out

def copy(x):
    out = empty_like(x)
    out[:] = x
    return out

def has_device_buffer(x):
    return x.ctypes.data in array_to_buffer

def get_device_buffer(x):
    return array_to_buffer[x.ctypes.data]


def randn(*shape, dtype=np.float64):
    out = empty(shape, dtype=dtype)
    _rng.standard_normal(shape, dtype=dtype, out=out)
    return out