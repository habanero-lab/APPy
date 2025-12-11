import numpy as np
import metalcompute as mc

device = mc.Device()

# Create a global generator instance
_rng = np.random.default_rng()

class SharedArray:
    def __init__(self, arr, buf):
        self.arr = arr        # actual numpy array
        self.buf = buf        # unified memory buffer (Metal)
        self.dev = device
    
    # auto unwrap to numpy array for any NumPy operation
    def __array__(self, dtype=None):
        if dtype is not None:
            return self.arr.astype(dtype)
        return self.arr
    
    def __repr__(self):
        return f"SharedArray(shape={self.arr.shape}, dtype={self.arr.dtype}):\n{str(self.arr)}"

        # left/right add/sub/mul/div
    def __add__(self, other):   return self._binary_op(other, np.add)
    def __radd__(self, other):  return self._binary_op(other, np.add)
    def __sub__(self, other):   return self._binary_op(other, np.subtract)
    def __rsub__(self, other):  return self._binary_op(other, lambda a, b: np.subtract(b, a))
    def __mul__(self, other):   return self._binary_op(other, np.multiply)
    def __rmul__(self, other):  return self._binary_op(other, np.multiply)
    def __truediv__(self, other):    return self._binary_op(other, np.divide)
    def __rtruediv__(self, other):   return self._binary_op(other, lambda a, b: np.divide(b, a))
    def __pow__(self, other):   return self._binary_op(other, np.power)
    def __rpow__(self, other):  return self._binary_op(other, lambda a, b: np.power(b, a))

    def _binary_op(self, other, op):
        if isinstance(other, SharedArray):
            return op(self.arr, other.arr)
        else:
            # Return a regular NumPy array
            return op(self.arr, other)


    @property
    def shape(self): return self.arr.shape
    @property
    def dtype(self): return self.arr.dtype
    @property
    def size(self):  return self.arr.size
    @property
    def __array_interface__(self):
        return self.arr.__array_interface__
    
    # make indexing behave transparently
    def __getitem__(self, idx):
        return self.arr[idx]
    def __setitem__(self, idx, val):
        self.arr[idx] = val

    def fill(self, val): 
        self.arr.fill(val)

def empty(shape, dtype=np.float64):
    dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    buf = device.buffer(n * dtype.itemsize)
    arr = np.frombuffer(buf, dtype=dtype, count=n).reshape(shape)
    return SharedArray(arr, buf)

def empty_like(x, dtype=None):
    return empty(x.shape, dtype or x.dtype)

def zeros(shape, dtype=np.float64):
    out = empty(shape, dtype=dtype)
    out.arr.fill(0)
    return out

def randn(*shape, dtype=np.float64):
    out = empty(shape, dtype=dtype)
    _rng.standard_normal(shape, dtype=dtype, out=out.arr)
    return out