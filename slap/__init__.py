from torch import arange
from .jit import jit

parallel = None
shared = None
dx = 128
dy = 1
dz = 1

def prange(start, end, step=1):
    return arange(start, end, step)

def syncthreads():
    return