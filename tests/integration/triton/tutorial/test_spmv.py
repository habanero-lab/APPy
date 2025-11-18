'''
Test sparse matrix vector multiplication.
'''

import numpy as np
import scipy.sparse as sp
import appy


@appy.jit(backend="triton", dump_code=True)
def kernel_appy(A_data, A_indptr, A_indices, x, M, N):
    y = np.empty(M, dtype=x.dtype)
    #pragma parallel for
    for i in range(M):
        start, end = A_indptr[i], A_indptr[i+1]
        s = 0.0
        #pragma simd
        for j in range(start, end):
            s += A_data[j] * x[A_indices[j]]
        y[i] = s
    return y


def test_csr():
    N = 1000
    A = sp.rand(N, N, density=0.01, format='csr')
    x = np.random.rand(N)
    y = kernel_appy(A.data, A.indptr, A.indices, x, N, N)
    y_ref = A @ x
    assert np.allclose(y, y_ref, atol=1e-6)