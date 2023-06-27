import torch
from torch import arange
from .jit import jit
#from .config import configs

parallel = None
shared = None
dx = 128
dy = 1
dz = 1

def prange(start, end, step=1):
    return arange(start, end, step)

def syncthreads():
    return

def max(a, axis=0):
    return torch.max(a, axis=axis)[0]