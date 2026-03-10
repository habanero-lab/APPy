'''
Maps Python/NumPy types to CUDA C types.
'''
import numpy as np

m = {
    # Built-in Python types
    'int': 'int',
    'float': 'float',
    'bool': 'bool',

    # NumPy scalar types
    'numpy.bool_': 'bool',
    'numpy.int8': 'int8_t',
    'numpy.uint8': 'uint8_t',
    'numpy.int16': 'int16_t',
    'numpy.uint16': 'uint16_t',
    'numpy.int32': 'int',
    'numpy.uint32': 'unsigned int',
    'numpy.float32': 'float',
    'numpy.float64': 'double',

    # NumPy array dtype strings
    'bool_': 'bool',
    'bool': 'bool',
    'int8': 'int8_t',
    'uint8': 'uint8_t',
    'int16': 'int16_t',
    'uint16': 'uint16_t',
    'int32': 'int',
    'uint32': 'unsigned int',
    'float32': 'float',
    'float64': 'double',
}

def get_cuda_type(val):
    if isinstance(val, str):
        return m[val]
    elif isinstance(val, (int, float, bool)):
        return m[type(val).__name__]
    elif isinstance(val, np.ndarray):
        return m[str(val.dtype)]
    elif isinstance(val, (np.bool_, np.int8, np.uint8, np.int16, np.uint16,
                          np.int32, np.uint32, np.float32, np.float64)):
        return m[type(val).__name__]
    else:
        raise NotImplementedError(f"Type mapping not implemented for value of type {type(val)}")

def is_arithmetic_scalar_cuda_type(ty):
    return ty in m.values()
