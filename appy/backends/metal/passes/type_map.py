'''
Maps Python types to Metal/CPP types.
'''
import numpy as np

m = {
    # Built-in Python type
    'int': 'int',
    'float': 'float',
    'bool': 'bool',

    # NumPy scalar type
    'numpy.bool_': 'bool',
    'numpy.int8': 'char',
    'numpy.uint8': 'uchar',
    'numpy.int16': 'short',
    'numpy.uint16': 'ushort',
    'numpy.int32': 'int',
    'numpy.uint32': 'uint',
    'numpy.int64': 'long',
    'numpy.uint64': 'ulong',
    'numpy.float32': 'float',
    'numpy.float64': 'float',

    # NumPy array dtype
    'bool_': 'bool',
    'bool': 'bool',
    'int8': 'char',
    'uint8': 'uchar',
    'int16': 'short',
    'uint16': 'ushort',
    'int32': 'int',
    'uint32': 'uint',
    'int64': 'long',
    'uint64': 'ulong',
    'float32': 'float',
    'float64': 'float'
}

def get_metal_type(val):
    if isinstance(val, str):
        return m[val]
    elif isinstance(val, (int, float, bool)):
        return m[type(val).__name__]
    elif isinstance(val, np.ndarray):
        return m[str(val.dtype)]
    elif isinstance(val, (np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)):
        return m[type(val).__name__]
    else:
        raise NotImplementedError(f"Type mapping not implemented for value of type {type(val)}")
    
def is_arithmetic_scalar_metal_type(ty):
    return ty in m.values()
