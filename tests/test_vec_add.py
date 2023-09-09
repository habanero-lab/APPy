import numpy as np
import torch
from torch import arange, zeros, empty
import appy
 
@appy.jit(auto_block=True)
def kernel(a, b, c, N):
    #pragma :N=>parallel
    c[:N] = a[:N] + b[:N]


for shape in [1024*128, 1024*1024, 1024*1024*2]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    ms = appy.utils.bench(lambda: a + b)
    print(f'torch: {ms} ms')

    
    for f in [kernel]:
        c = torch.zeros_like(a)
        f(a, b, c, N)
        assert(torch.allclose(c, a+b))
        ms = appy.utils.bench(lambda: f(a, b, c, N))
        print(f'kernel: {ms} ms')

@appy.jit(tune={'APPY_BLOCK': [128, 256, 512, 1024]})
def kernel(a, b, c, N):
    #pragma parallel
    for _top_var_0 in range(0, N, APPY_BLOCK):
        _top_var_0 = vidx(_top_var_0, APPY_BLOCK, N)
        c[_top_var_0] = a[_top_var_0] + b[_top_var_0]