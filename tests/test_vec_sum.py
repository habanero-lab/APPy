import torch
import bmp

#@bmp.jit(tune=['BLOCK'])
def kernel(a, b, BLOCK):
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel reduction(+:b)
        s = torch.sum(a[i:i+BLOCK])
        b[0] += s 
        

for shape in [1024*128, 1024*1024]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        b = torch.zeros(1, device='cuda', dtype=torch.float32)
        f(a, b, 128)
        assert(torch.allclose(b, torch.sum(a)))
