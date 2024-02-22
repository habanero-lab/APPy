import torch
import cupy
from .jit import jit
from .utils import *

torch.set_default_device('cuda')
tensorlib = torch  # Possible options are: torch, cupy

# Array creation functions
def empty(size, dtype):
    return tensorlib.empty(size, dtype=dtype)

def zeros(size, dtype):
    return tensorlib.zeros(size, dtype=dtype)

def empty_like(a):
    return tensorlib.empty_like(a)

def zeros_like(a):
    return tensorlib.zeros_like(a)

def randn(*size, dtype='float64'):
    a = cupy.random.randn(*size).astype(dtype)
    if tensorlib == torch:
        a = torch.as_tensor(a)
    return a

# Math functions
def sum(a, axis=0):
    return tensorlib.sum(a, axis)

def dot(a, b):
    return tensorlib.dot(a, b)

def mv(a, b):
    return tensorlib.mv(a, b)

def minimum(a, b):
    return tensorlib.minimum(a, b)

def where(*args):
    return tensorlib.where(*args)

def mean(args, axis=0):
    return tensorlib.mean(args, axis)

def sqrt(a):
    return tensorlib.sqrt(a)

def exp(a):
    return tensorlib.exp(a)

def log(a):
    return tensorlib.log(a)

def max(a, axis=0):
    if tensorlib == torch:
        return tensorlib.max(a, axis=axis)[0]
    elif tensorlib == cupy:
        return tensorlib.max(a, axis=axis)

# Special functions
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