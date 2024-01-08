import torch
import appy
import utils

@appy.jit
def kernel_appy(a, b):
    N = a.shape[0]
    c = torch.empty_like(a)
    #pragma parallel for simd 
    for i in range(N):
        c[i] = a[i] + b[i]
    return c

def kernel_torch(a, b):
    return a + b

def test():
    inputs = utils.get_1d_tensors_assorted_shapes(10, tuple_size=2)
    for a, b in inputs:
        c_ref = kernel_torch(a, b)
        print(f'N: {a.shape[0]}, dtype: {a.dtype}')
        for f in [kernel_torch, kernel_appy]:
            c = f(a, b)
            assert(appy.utils.allclose(c, c_ref, atol=1e-6))
            ms = appy.utils.bench(lambda: f(a, b))
            print(f'{f.__name__}: {ms:.4f} ms')


if __name__ == '__main__':
    #utils.get_1d_tensors_assorted_shapes(10)
    test()