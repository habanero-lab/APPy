import torch
from slap import jit
from slap.utils import bench
from torch import arange, zeros, empty, sum

nclusters = 100

@jit
def mykernel(x, labels, centers, M, N, Bj):
    #pragma parallel reduction(centers) 
    for i in range(M):
        #pragma parallel
        for j in range(0, N, Bj):  
            label = labels[i]
            centers[label,j:j+Bj] += x[i,j:j+Bj]

#@jit
def mykernel1(x, labels, centers, M, N, Bj):
    #pragma parallel reduction(centers) 
    for i in range(M):  
        #pragma par_dim(:N:Bj)
        centers[labels[i], :N] += x[i, :N]

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

        for f in [torch_kernel1, mykernel]:
            centers = torch.zeros([nclusters, N], device='cuda', dtype=torch.float32)
            BLOCK = 128 * 1
            f(X, labels, centers, M, N, BLOCK)
            assert(torch.allclose(centers_ref, centers, atol=1e-2))
            ms = bench(lambda: f(X, labels, centers, M, N, BLOCK))
            print(f'{f.__name__}: {ms:.4f} ms')
            
if __name__ == '__main__':
    test1()
