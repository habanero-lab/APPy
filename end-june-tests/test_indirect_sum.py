import torch
import triton
import triton.language as tl
import slap 
from torch import arange, zeros, empty, sum

nclusters = 200

@slap.jit(dump_code=0)
def slap_kernel(x, labels, centers, M, N, Bj):
    for i in range(M):  #pragma parallel reduction(centers) 
        for j in range(0, N, Bj):  #pragma parallel
            label = labels[i]
            centers[label,j:j+Bj] += x[i,j:j+Bj]

@slap.jit(dump_code=0)
def slap_kernel1(x, labels, centers, M, N, Bj):
    for i in range(M):  #pragma parallel reduction(centers) 
        for j in range(N):  #pragma parallel block(128)
            label = labels[i]
            centers[label,j] += x[i,j]

def torch_kernel(x, labels, centers, M, N, Bj=None):
    for i in range(M):
        label = labels[i]
        centers[label, :] += x[i, :]

def torch_kernel1(x, labels, centers, M, N, Bj=None):
    for i in range(nclusters):
        filtered_x = x[labels == i]
        centers[i] = sum(filtered_x, axis=0)

def test1():
    for M, N in [(60000, 768), (1024*128, 128)]:
        print(f'M: {M}, N: {N}')
        X = torch.randn(M, N, device='cuda', dtype=torch.float32)
        labels = torch.randint(low=0, high=nclusters, size=(M,), device='cuda', dtype=torch.int32)
        centers_ref = torch.zeros([nclusters, N], device='cuda', dtype=torch.float32)
        torch_kernel(X, labels, centers_ref, M, N)

        for f in [torch_kernel1, slap_kernel1]:
            centers = torch.zeros([nclusters, N], device='cuda', dtype=torch.float32)
            BLOCK = 128 * 1
            f(X, labels, centers, M, N, BLOCK)
            assert(torch.allclose(centers_ref, centers, atol=1e-2))
            ms, _, _ = triton.testing.do_bench(lambda: f(X, labels, centers, M, N, BLOCK))
            print(f'{f.__name__}: {ms:.4f} ms')
            
