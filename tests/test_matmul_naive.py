import torch

def kernel(a, b, c, Bi, Bj, Bk):
    for i in range(0, a.shape[0], Bi):  #pragma parallel
        for j in range(0, b.shape[-1], Bj):  #pragma parallel
            c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
            for k in range(a.shape[-1]):
                c_block[:, :] += a[i:i+Bi, k] * b[k, j:j+Bj]
            c[i:i+Bi, j:j+Bj] = c_block

def kernel1(a, b, c, Bi, Bj, Bk):
    for i in range(0, a.shape[0], Bi):  #pragma parallel
        for j in range(0, b.shape[-1], Bj):  #pragma parallel
            c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
            a_block = bmp.shared([Bi, Bk], dtype=a.dtype)
            b_block = bmp.shared([Bk, Bj], dtype=b.dtype)
            a_block[:,:] = a[i:i+Bi, k:k+Bk]
            
            
            for k in range(0, a.shape[-1], Bk):
                c_block[:, :] += a[i:i+Bi, k] * b[k, j:j+Bj]
            c[i:i+Bi, j:j+Bj] = c_block

