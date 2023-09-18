import torch
import appy

@appy.jit
def kernel_block_oriented(a, b, N, BLOCK=512):
    '''
    The output array must be zero-initialized in order to do parallel reduction.
    '''
    #pragma parallel
    for i in range(0, N, BLOCK):  
        vi = appy.vidx(i, BLOCK, bound=N)
        #pragma atomic
        b[0] += torch.sum(a[vi])

#@appy.jit(auto_block=True)     # TODO: parallel reduction in TEs are not supported yet 
def kernel_tensor_oriented(a, b, N):
    #pragma :N=>parallel,reduce(sum:b)
    b[0] = torch.sum(a[:N])

def kernel_torch(a, b, N):
    b[0] = torch.sum(a)

def test():
    for dtype in [torch.float32, torch.float64]:
        for N in [1024*1024, 4*1024*1024, 4*1024*1024-1]:
            print(f'N: {N}, dtype: {dtype}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b_ref = torch.empty(1, device='cuda', dtype=dtype)
            kernel_torch(a, b_ref, N)
            
            for f in [kernel_torch, kernel_block_oriented]:
                b = torch.zeros(1, device='cuda', dtype=dtype)
                f(a, b, N)
                assert(appy.utils.allclose(b, b_ref, atol=1e-3))
                ms = appy.utils.bench(lambda: f(a, b, N))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test()