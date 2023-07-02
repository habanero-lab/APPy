import torch
from slap import jit, max
from torch import arange, zeros, empty, sum, maximum, add, exp

torch.set_default_device('cuda')

def mykernel(a, b, c, N):
    #pragma parallel
    c[:N] = a[:N] + b[:N]

def mykernel(a, b, c, N):
    #pragma parallel reduction(c)
    c[0] = sum(a[:N] * b[:N])

    # Fusion is implied for a single slicing statement
    # Different slicing statements can be fused using pragma fuse

def mykernel(a, b, c, M, N, K, BM, BN):
    #pragma parallel
    for i in range(0, M, BM):
        #pragma parallel
        for j in range(0, N, BN):
            c[i:i+BM, j:j+BN] = a[i:i+BM, :K] @ b[:K, j:j+BN]

def gelu(x, y, N):
    #pragma parallel
    y[:N] = 0.5 * x[:N] * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x[:N] + 0.044715 * x[:N] ** 3)))

# Long slices are by default blocked, and run sequentially