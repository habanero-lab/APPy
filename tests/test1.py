import numpy as np
import appy

@appy.jit(backend="triton", dry_run=True)
def kernel_prange(a, b):
    c = np.empty_like(a)
    for i in appy.prange(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

a = np.ones(4)
b = np.ones(4)
res = kernel_prange(a, b)  # Runs Python fallback, writes Triton code to .appy_kernels/kernel_prange.py

print(res)