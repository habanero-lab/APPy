import torch
import bmp

#@bmp.jit(tune=['BLOCK'])
def add(a, b, c, BLOCK):
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel
        idx = range(i, min(i+BLOCK, a.shape[0]))
        c[idx] = a[idx] + b[idx]
        

for shape in [1024, 1024-1]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)

    for f in [add]:
        c = torch.zeros_like(a)
        f(a, b, c, 128)
        assert(torch.allclose(c, a+b))
