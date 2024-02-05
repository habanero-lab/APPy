import torch
import appy
import numba

@appy.jit
def kernel_block_oriented(a, b, c, N, BLOCK=256):
    #pragma parallel
    for i in range(0, N, BLOCK):  
        vi = appy.vidx(i, BLOCK, bound=N)
        c[vi] = a[vi] + b[vi]

@appy.jit
def kernel_loop_no_simd(a, b, c, N):
    #pragma parallel
    for i in range(N):  
        c[i] = a[i] + b[i]

@numba.jit(nopython=True)
def kernel_numba(a, b, c, N):
    for i in numba.prange(N):
        c[i] = a[i] + b[i]

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

            a_np = a.cpu().numpy()
            b_np = b.cpu().numpy()
            c_np = c_ref.cpu().numpy()
            np_ms = appy.utils.bench(lambda: a_np + b_np)
            nb_ms = appy.utils.bench(lambda: kernel_numba(a_np, b_np, c_np, N))
            print(f'numpy: {np_ms:.4f} ms')
            print(f'numba: {nb_ms:.4f} ms')
            
            for f in [kernel_torch, kernel_loop_no_simd, kernel_block_oriented, kernel_tensor_oriented]:
                c = torch.empty_like(a)
                f(a, b, c, N)
                assert(torch.allclose(c, c_ref))
                ms = appy.utils.bench(lambda: f(a, b, c, N))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test_vec_add()



'''
@appy.jit
def loop_kernel(a, b, c, N):
    #pragma parallel
    for i in range(N):  
        c[i] = a[i] + b[i]

@appy.jit
def loop_kernel(a, b, c, N, BN=256):
    #pragma parallel
    for i in range(0, N, BN):  
        i = appy.vidx(i, BN, bound=N)
        c[i] = a[i] + b[i]

@appy.jit(auto_block=True)
def tensor_kernel(a, b, c, N):
    #pragma :N=>parallel
    c[:N] = a[:N] + b[:N] 
'''