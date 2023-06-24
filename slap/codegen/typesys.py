class Tensor(object):
    def __init__(self, dtype, ndim, shape=None):
        self.dtype = dtype
        self.ndim = ndim
        self.shape = shape

    