import torch
import appy

#@appy.jit(tune=['Bi', 'Bj', 'Bk'])
def kernel(a, b, c, Bi, Bj):
    for i in range(a.shape[0]):  #pragma parallel
        s = 0
        for j in range(0, b.shape[-1], Bj): 
            s += torch.sum(a[i,j:j+Bj] * b[j:j+Bj])
        c[i] = s


def kernel(a, b, c, Bi, Bj):
    for i in range(a.shape[0]):  #pragma parallel
        buf = torch.zeros([Bj], device=a.device, dtype=a.dtype)
        for j in range(0, b.shape[-1], Bj): 
            buf += a[i,j:j+Bj] * b[j:j+Bj]
        c[i] = torch.sum(buf)


def kernel_block_i(a, b, c, Bi, Bj):
    for i in range(0, a.shape[0], Bi):  #pragma parallel
        c_block = torch.zeros([Bi], device=a.device, dtype=a.dtype)
        for j in range(0, b.shape[-1], Bj): 
            c_block += torch.sum(a[i:i+Bi,j:j+Bj] * b[j:j+Bj][None,:])
        c[i:i+Bi] = c_block



for M, N in [(1024, 1024)]:
    print(f'M: {M}, N: {N}')
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        c = torch.zeros(M, device='cuda', dtype=torch.float32)
        kernel(a, b, c, 32, 32, 32)
        assert(torch.allclose(c, a@b, atol=1e-3))
