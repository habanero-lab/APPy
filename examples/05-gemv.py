import appy
# import cupy
# appy.tensorlib = cupy

@appy.jit(auto_simd=True)
def kernel_appy(alpha, A, x):
    M, N = A.shape
    y = appy.empty(M, dtype=A.dtype)
    y[:M] = alpha * appy.mv(A[:M, :N], x[:N])
    return y


def kernel_lib(alpha, A, x):
    y = alpha * A @ x
    return y
    

def test():
    for N in [1000, 4000, 8000]:
        A = appy.randn(N, N)
        x = appy.randn(N)
        alpha = 1.0
        y_ref = kernel_lib(alpha, A, x)
        for f in [kernel_lib, kernel_appy]:
            y = f(alpha, A, x)
            assert appy.utils.allclose(y, y_ref, atol=1e-6)
            ms = appy.utils.bench(lambda: f(alpha, A, x))
            print(f"{f.__name__}: {ms:.4f} ms")


if __name__ == "__main__":
    test()
