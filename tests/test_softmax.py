import torch
import slap

def kernel(a, b):
    for i in range(a.shape[0]):  #pragma parallel 
        row = a[i,:]
        m = row.max()
        e = (row - m).exp()
        s = e.sum()
        b[i:] = e / s

for M in [1024]:
    N = 1024
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        b = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        kernel(a, b) 
        assert(torch.allclose(b, torch.softmax(a, dim=1), atol=1e-3))
