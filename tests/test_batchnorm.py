import torch
import appy
from appy import Dx

def kernel(a, b, M, N, Bi):
    mean = torch.empty(N, device=a.device, dtype=a.dtype)
    for i in range(0, M, Bi):  #pragma parallel reduction(+:mean)
        for j in range(0, N, Dx):  #pragma parallel 
            c = sum(a[i:i+Bi, j:j+Dx], axis=0)
            mean[j:j+Dx] += c

    var = torch.empty(N, device=a.device, dtype=a.dtype)
    for i in range(0, M, Bi):  #pragma parallel reduction(+:mean)
        for j in range(0, N, Dx):  #pragma parallel 
            d = a[i:i+Bi, j:j+Dx] - mean[j:j+Dx]
            c = sum(d, axis=0)
            var[j:j+Dx] += c



for M in [1024]:
    N = 1024
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        b = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        kernel(a, b) 
        assert(torch.allclose(b, torch.softmax(a, dim=1), atol=1e-3))
