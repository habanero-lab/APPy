import torch
import appy

@appy.jit
def kernel_block_oriented(a, b, c, N, BLOCK=256):
    #pragma parallel
    for i in range(0, N, BLOCK):  
        vi = vidx(i, BLOCK, bound=N)
        c[vi] = a[vi] + b[vi]

@appy.jit(auto_block=True)
def kernel_tensor_oriented(a, b, c, N):
    #pragma :N=>parallel
    c[:N] = a[:N] + b[:N] 

def kernel_torch(a, b, c, N):
    torch.add(a, b, out=c)

def test_vec_add():
    for dtype in [torch.float32, torch.float64]:
        for N in [1024*1024, 4*1024*1024, 4*1024*1024-1]:
            print(f'N: {N}, dtype: {dtype}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b = torch.randn(N, device='cuda', dtype=dtype)
            c_ref = torch.randn(N, device='cuda', dtype=dtype)
            kernel_torch(a, b, c_ref, N)
            
            for f in [kernel_torch, kernel_block_oriented, kernel_tensor_oriented]:
                c = torch.empty_like(a)
                f(a, b, c, N)
                assert(torch.allclose(c, c_ref))
                ms = appy.utils.bench(lambda: f(a, b, c, N))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test_vec_add()