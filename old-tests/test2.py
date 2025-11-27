import numpy as np
import appy

@appy.jit(backend="ptx", dry_run=0, dump_code=True)
def kernel_prange(a, b):
    c = np.empty_like(a)
    for i in appy.prange(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

a = np.ones(4).astype(np.float64)
b = np.ones(4).astype(np.float64)
res = kernel_prange(a, b)  
print(res)