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
    array_to_buffer[arr.ctypes.data] = (buf, device)
    return arr

def empty_like(x, dtype=None):
    return empty(x.shape, dtype or x.dtype)

def zeros(shape, dtype=np.float64):
    out = empty(shape, dtype=dtype)
    return out

def randn(*shape, dtype=np.float64):
    out = empty(shape, dtype=dtype)
    _rng.standard_normal(shape, dtype=dtype, out=out)
    return out