import torch
import bmp

def kernel(a, b):
    mean = torch.empty(a.shape[1], device=a.device, dtype=a.dtype)
    for j in range(0, a.shape[1], Bj):  #pragma parallel 
        for i in range(0, a.shape[0], Bi):  #pragma parallel reduction(+:mean)
            data = a[i:i+Bi, j:j+Bj]
            c = data.sum(axis=0)
            mean[j:j+Bj] += c

    var = 

for M in [1024]:
    N = 1024
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        b = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        kernel(a, b) 
        assert(torch.allclose(b, torch.softmax(a, dim=1), atol=1e-3))
