import numpy as np
import appy

@appy.jit(backend="triton", dry_run=0)
def kernel_prange(a, b):
    c = np.empty_like(a)
    #pragma parallel for simd
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

a = np.ones(8)
b = np.ones(8)
res = kernel_prange(a, b)  # Runs Python fallback, writes Triton code to .appy_kernels/kernel_prange.py
print(res)