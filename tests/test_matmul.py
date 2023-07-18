import torch
import appy
from appy import parallel, shared

def kernel(a, b, c, Bi, Bj, Bk):
    for i in range(0, a.shape[0], Bi):  #pragma parallel
        for j in range(0, b.shape[-1], Bj):  #pragma parallel
            c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
            for k in range(0, a.shape[-1]):
                c_block[:, :] += a[i:i+Bi, k] * b[k, j:j+Bj]
            c[i:i+Bi, j:j+Bj] = c_block


def kernel(a, b, c, Bi:parallel, Bj:parallel, Bk):
    for i in range(0, a.shape[0], Bi):  #pragma parallel
        for j in range(0, b.shape[-1], Bj):  #pragma parallel
            c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
            a_block:shared = torch.zeros([Bi, Bk], device=a.device, dtype=a.dtype)
            b_block:shared = torch.zeros([Bk, Bj], device=a.device, dtype=a.dtype)
            for k in range(0, a.shape[-1], Bk):
                # Load data to shared memory
                appy.syncthreads()
                a_block[:,:] = a[i:i+Bi, k:k+Bj]
                b_block[:,:] = b[k:k+Bi, j:j+Bj]
                appy.syncthreads()

                for kk in range(k, k+Bk):
                    c_block[:, :] += a_block[:Bi, kk] * b_block[kk, :Bj]
                
            c[i:i+Bi, j:j+Bj] = c_block


#@appy.jit(tune=['Bi', 'Bj', 'Bk'])
def kernel(a, b, c, Bi, Bj, Bk):
    for i in range(0, a.shape[0], Bi):  #pragma parallel
        for j in range(0, b.shape[-1], Bj):  #pragma parallel
            c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
            for k in range(0, a.shape[-1], Bk):
                c_block[:, :] += a[i:i+Bi, k:k+Bk] @ b[k:k+Bk, j:j+Bj]
            c[i:i+Bi, j:j+Bj] = c_block


for M, N, K in [(1024, 1024, 1024)]:
    print(f'M: {M}, N: {N}, K: {K}')
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)

    for f in [kernel]:
        c = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        kernel(a, b, c, 32, 32, 32)
        assert(torch.allclose(c, a@b, atol=1e-3))
