import torch
from torch import arange
from .jit import jit
from .utils import *
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

def sum(a, axis=0):
    return torch.sum(a, axis)

def empty(size, dtype):
    return torch.empty(size, dtype=dtype, device='cuda')

def zeros(size, dtype):
    return torch.zeros(size, dtype=dtype, device='cuda')

def empty_like(a):
    return torch.empty_like(a)

def zeros_like(a):
    return torch.zeros_like(a)

def dot(a, b):
    return torch.dot(a, b)

def mv(a, b):
    return torch.mv(a, b)

def minimum(a, b):
    return torch.minimum(a, b)

def where(*args):
    return torch.where(*args)

def mean(args, axis=0):
    return torch.mean(args, axis)

def sqrt(a):
    return torch.sqrt(a)

def step(start, stepsize, bound=None):
    if bound:
        r = slice(start, min(bound, start+stepsize))
    else:
        r = slice(start, start+stepsize)
    return r

def debug_barrier():
    pass

def atomic_add(a, offset, b):
    a[offset] += b

vindex = step
vidx = step

def get_matmul_configs(BM, BN, BK):
    return [
        {BM: 128, BN: 256, BK: 32, 'num_stages': 3, 'num_warps': 8},
        {BM: 256, BN: 128, BK: 32, 'num_stages': 3, 'num_warps': 8},
        {BM: 256, BN: 64, BK: 32, 'num_stages': 4, 'num_warps': 8},
        {BM: 64, BN: 256, BK: 32, 'num_stages': 4, 'num_warps': 8},
        {BM: 128, BN: 128, BK: 32, 'num_stages': 4, 'num_warps': 8},


        {BM: 256, BN: 64, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 64, BN: 256, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 128, BN: 128, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 128, BN: 64, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 64, BN: 128, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 128, BN: 32, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 64, BN: 32, BK: 32, 'num_stages': 5, 'num_warps': 2},
    ]