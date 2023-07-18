import torch
import appy

#@appy.jit(tune=['Bi', 'Bj', 'Bk'])
def kernel(a, b, c, Bi, Bj, Bk):
    for z in range(a.shape[0]):  #pragma parallel
        for i in range(0, a.shape[1], Bi):  #pragma parallel
            for j in range(0, b.shape[-1], Bj):  #pragma parallel
                c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
                for k in range(0, a.shape[-1], Bk):
                    c_block[:, :] += a[z, i:i+Bi, k:k+Bk] @ b[z, k:k+Bk, j:j+Bj]
                c[z, i:i+Bi, j:j+Bj] = c_block


for M, N, K in [(1024, 1024, 1024)]:
    Z = 4
    print(f'M: {M}, N: {N}, K: {K}')
    a = torch.randn(Z, M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(Z, K, N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        c = torch.zeros(Z, M, N, device='cuda', dtype=torch.float32)
        kernel(a, b, c, 32, 32, 32) 
        assert(torch.allclose(c, a@b, atol=1e-3))
