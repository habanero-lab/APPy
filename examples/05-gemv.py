import torch
import appy
from benchmark_utils import allclose, bench

torch.set_default_device('cuda')

@appy.jit(dump_code=True)
def kernel_appy(alpha, A, x):
    y = torch.empty(A.shape[0], dtype=x.dtype)
    #pragma parallel for
    for i in range(A.shape[0]):
        yi = 0.0
        #pragma simd
        for j in range(A.shape[1]):
            yi += alpha * A[i, j] * x[j]
        y[i] = yi
    return y


def kernel_lib(alpha, A, x):
    y = alpha * A @ x
    return y
    

def test():
    for N in [1000, 4000, 8000]:
        A = torch.randn(N, N, dtype=torch.float64)
        x = torch.randn(N, dtype=torch.float64)
        alpha = 1.0
        y_ref = kernel_lib(alpha, A, x)
        for f in [kernel_lib, kernel_appy]:
            y = f(alpha, A, x)
            assert allclose(y, y_ref, atol=1e-6)
            ms = bench(lambda: f(alpha, A, x))
            print(f"{f.__name__}: {ms:.4f} ms")


if __name__ == "__main__":
    test()
