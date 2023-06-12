import torch
from slap import parallel
from torch import max, sum, exp

def kernel_op(a):
    b = exp(a - max(a))
    return b / sum(b, axis=1)


def kernel_python(a, b, M, N):
    for i in range(M):
        m = MAX_FLOAT
        for j in range(N):
            m = max(m, a[i,j])
        
        for j in range(N):
            e[i,j] = torch.exp(a[i,j] - m)

        s = 0
        for j in range(N):
            s += e[i,j]

        for j in range(N):
            b[i,j] = a[i,j] / s

def kernel(a, b, N, BLOCK: parallel):
    for i in range(N):  #pragma parallel 
        row = a[i,:BLOCK]
        m = row.max()
        e = (row - m).exp()
        s = e.sum()
        b[i,:BLOCK] = e / s

for M in [1024]:
    N = 1024
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        b = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        kernel(a, b) 
        assert(torch.allclose(b, torch.softmax(a, dim=1), atol=1e-3))
