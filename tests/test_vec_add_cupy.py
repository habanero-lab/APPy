import cupy
import appy
appy.tensorlib = cupy

@appy.jit
def kernel_appy(a, b):
    N = a.shape[0]
    c = appy.empty_like(a)
    #pragma parallel for simd
    for i in range(N):
        c[i] = a[i] + b[i]
    return c


def kernel_lib(a, b):
    return a + b

for N in [10000, 100000, 1000000, 10000000]:
    print('array size:', N)
    a = cupy.random.randn(N)
    b = cupy.random.randn(N)
    c_ref = kernel_lib(a, b)
    c = kernel_appy(a, b)
    # Check for results correctness
    assert(cupy.allclose(c, c_ref))

    # Measure performance
    for f in [kernel_lib, kernel_appy]:
        ms = appy.utils.bench(lambda: f(a, b))
        print(f"{f.__name__}: {ms:.4f} ms")