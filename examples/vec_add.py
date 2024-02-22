import appy


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


def test():
    for N in [10000, 100000, 1000000, 10000000]:
        a = appy.randn(N)
        b = appy.randn(N)
        c_ref = kernel_lib(a, b)
        print(f"N: {a.shape[0]}, dtype: {a.dtype}")
        for f in [kernel_lib, kernel_appy]:
            c = f(a, b)
            assert appy.utils.allclose(c, c_ref, atol=1e-6)
            ms = appy.utils.bench(lambda: f(a, b))
            print(f"{f.__name__}: {ms:.4f} ms")


if __name__ == "__main__":
    test()
