import torch
import appy
import utils

@appy.jit
def kernel_appy(a):
    # Zero-initialize the output array
    b = torch.zeros(1, device=a.device, dtype=a.dtype)
    N = a.shape[0]
    #pragma parallel for simd reduction
    for i in range(N): 
        #pragma atomic
        b[0] += a[i]
    return b

def kernel_torch(a):
    return torch.sum(a)

def test():
    inputs = utils.get_1d_tensors_assorted_shapes(10)
    for a, in inputs:
        b_ref = kernel_torch(a)
        print(f'N: {a.shape[0]}, dtype: {a.dtype}')
        for f in [kernel_torch, kernel_appy]:
            b = f(a)
            assert(appy.utils.allclose(b, b_ref, atol=1e-1))
            ms = appy.utils.bench(lambda: f(a))
            print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test()