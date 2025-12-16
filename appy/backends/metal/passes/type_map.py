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
    'numpy.bool': 'bool',
    'numpy.int8': 'char',
    'numpy.uint8': 'uchar',
    'numpy.int16': 'short',
    'numpy.uint16': 'ushort',
    'numpy.int32': 'int',
    'numpy.uint32': 'uint',
    'numpy.float32': 'float',

    # NumPy array dtype
    'bool': 'bool',
    'int8': 'char',
    'uint8': 'uchar',
    'int16': 'short',
    'uint16': 'ushort',
    'int32': 'int',
    'uint32': 'uint',
    'float32': 'float'
}

def get_metal_type(val):
    if isinstance(val, str):
        return m[val]
    elif isinstance(val, (int, float, bool)):
        return m[type(val).__name__]
    elif isinstance(val, np.ndarray):
        return m[str(val.dtype)]
    elif isinstance(val, (np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float32)):
        return m[type(val).__name__]
    else:
        raise NotImplementedError(f"Type mapping not implemented for value of type {type(val)}")
    
def is_arithmetic_scalar_metal_type(ty):
    return ty in m.values()