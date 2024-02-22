import appy

# APPy tensors are `torch` tensors by default. Uncomment the 
# following lines to make APPy work with (and return) `cupy` tensors.
#import cupy
#appy.tensorlib = cupy

@appy.jit  # Comment this line to run the function in the Python interpter (debug mode)
def kernel_appy(a):
    ## Zero-initialize the output array
    b = appy.zeros(1, dtype=a.dtype)
    N = a.shape[0]
    #pragma parallel for simd
    for i in range(N): 
        #pragma atomic
        b[0] += a[i]
    return b


def kernel_lib(a):
    return a.sum()


def test():
    # TODO: torch.float32 has large results difference
    for N in [10000, 100000, 1000000, 10000000]:
        a = appy.randn(N, dtype='float32')
        c_ref = kernel_lib(a)
        print(f"N: {a.shape[0]}, dtype: {a.dtype}")
        for f in [kernel_lib, kernel_appy]:
            c = f(a)
            assert appy.utils.allclose(c, c_ref, atol=1e-6)
            ms = appy.utils.bench(lambda: f(a))
            print(f"{f.__name__}: {ms:.4f} ms")

if __name__ == '__main__':
    test()