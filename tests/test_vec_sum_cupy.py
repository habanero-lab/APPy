import cupy
import appy
import utils

appy.tensorlib = cupy

@appy.jit
def kernel_appy(a):
    # Zero-initialize the output array
    b = appy.zeros(1, dtype=a.dtype)
    N = a.shape[0]
    #pragma parallel for simd
    for i in range(N): 
        #pragma atomic
        b[0] += a[i]
    return b

def kernel_lib(a):
    return cupy.sum(a)

def test():
    # TODO: torch.float32 has large results different
    inputs = utils.get_random_1d_tensors(5, dtypes=['float64'])
    for a in inputs:
        a = cupy.asarray(a)
        b_ref = kernel_lib(a)
        print(f'N: {a.shape[0]}, dtype: {a.dtype}')
        for f in [kernel_lib, kernel_appy]:
            b = f(a)
            assert(appy.utils.allclose(b, b_ref, atol=1e-6))
            ms = appy.utils.bench(lambda: f(a))
            print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test()