
class Tensor(object):
    def __init__(self, dtype, ndim, shape=None):
        self.dtype = dtype
        self.ndim = ndim
        self.shape = shape

    def get_tl_dtype(self):
        return get_tl_dtype_from_str(str(self.dtype))

    def __str__(self):
        return f'Tensor({self.dtype}, {self.ndim})'    

class Constant(object):
    def __init__(self, dtype, value):
        self.dtype = dtype
        self.value = value


def build_type_from_value(v):
    if f"{type(v).__module__}.{type(v).__name__}" == 'torch.Tensor':
    #if isinstance(v, torch.Tensor):
        ty = Tensor(v.dtype, v.dim())
    elif isinstance(v, int) or isinstance(v, float) or f"{type(v).__module__}.{type(v).__name__}" == 'torch.dtype':
        ty = Constant(type(v), v)
    return ty
   
def get_tl_dtype_from_str(s):
    return 'tl.' + s.replace('torch.', '')