import appy
import inspect

def f0(a, b):
    c = torch.empty_like(a)
    #pragma parallel for simd
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

def f1(a):
    ## Zero-initialize the output array
    b = torch.zeros(1, dtype=a.dtype)
    #pragma parallel for simd
    for i in range(a.shape[0]): 
        #pragma atomic
        b[0] += a[i]
    return b

def f2(A_data, A_indptr, A_indices, x, M, N):
    y = torch.empty(M, dtype=x.dtype)
    #pragma parallel for
    for i in range(M):
        y[i] = 0.0
        #pragma simd
        for j in range(A_indptr[i], A_indptr[i+1]):
            y[i] += A_data[j] * x[A_indices[j]]
    return y

def f3(A, B):
    M, N = A.shape
    for t in range(1, 10):
        #pragma 1:M-1=>parallel 1:N-1=>parallel
        B[1:M-1, 1:N-1] = 0.2 * (A[1:M-1, 1:N-1] + A[1:M-1, :N-2] + A[1:M-1, 2:N] +
                                A[2:M, 1:N-1] + A[0:M-2, 1:N-1])
        #pragma 1:M-1=>parallel 1:N-1=>parallel
        A[1:M-1, 1:N-1] = 0.2 * (B[1:M-1, 1:N-1] + B[1:M-1, :N-2] + B[1:M-1, 2:N] +
                                B[2:M, 1:N-1] + B[0:M-2, 1:N-1])
    return A, B

def f4(A_indptr, A_indices, A_data, A_shape, B_indptr, B_indices, B_data, B_shape, C_indptr, C_indices, C_data, C_shape):
    __dB = empty((A_shape[0], A_shape[1]))
    __v2 = empty((A_shape[0], A_shape[1]))
    __ret = empty((A_shape[0], A_shape[1]))
    __dB_shape_1 = __dB.shape[1]
    __dB_shape_0 = __dB.shape[0]
    __v2_shape_1 = __v2.shape[1]
    __v2_shape_0 = __v2.shape[0]
    __ret_shape_1 = __ret.shape[1]
    __ret_shape_0 = __ret.shape[0]
    #pragma parallel for
    for i in range(0, __dB_shape_0, 1):
        #pragma simd
        for j in range(0, __dB_shape_1, 1):
            __dB[i, j] = 0
        #pragma simd
        for __pB_i in range(B_indptr[i], B_indptr[i + 1], 1):
            j = B_indices[__pB_i]
            __dB[i, j] = __dB[i, j] + B_data[__pB_i] * 1 # target_indices: ['i', 'j']
        #pragma simd
        for j in range(0, __v2_shape_1, 1):
            __v2[i, j] = __dB[i, j]
        for __pA_i in range(A_indptr[i], A_indptr[i + 1], 1):
            j = A_indices[__pA_i]
            __v2[i, j] = __v2[i, j] + A_data[__pA_i] # target_indices: ['i', 'j']
        #pragma simd
        for j in range(0, __ret_shape_1, 1):
            __ret[i, j] = __v2[i, j]
        for __pC_i in range(C_indptr[i], C_indptr[i + 1], 1):
            j = C_indices[__pC_i]
            __ret[i, j] = __ret[i, j] + C_data[__pC_i] # target_indices: ['i', 'j']
    return __ret

def f5(n, alpha, dx, dt, u, u_tmp):
    r = alpha * dt / (dx * dx)
    r2 = 1.0 - 4.0 * r

    #pragma parallel for                                                                                                              
    for i in range(n):
        #pragma simd                                                                                                                  
        for j in range(n):
            u_tmp[i, j] = (
                r2 * u[i, j] + 
                r * torch.where(i<n-1, u[i+1, j], 0.0) +
                r * torch.where(i>0, u[i-1, j], 0.0) +
                r * torch.where(j<n-1, u[i, j+1], 0.0) +
                r * torch.where(j>0, u[i, j-1], 0.0)
            )


def f6(a, b):
    '''
    The `to` and `from` clause are used to move array data between host 
    and device (from the viewpoint of the host). Scalar variables are
    passed to the GPU kernel with firstprivate property, unless explicitly
    specified in the `global` clause (see the next example). 
    '''
    c = np.empty_like(a)
    #pragma parallel for simd to(a,b) tofrom(c)
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

def f7(a):
    '''
    The global clause moves a scalar variable to the GPU global memory,
    and copies it back after the parallel region exits.
    '''
    b = 0.0
    #pragma parallel for simd to(a) global(b)
    for i in range(a.shape[0]): 
        #pragma atomic
        b += a[i]
    return b

def f8(a):
    '''
    Use appy.to_gpu() and appy.from_gpu() to move array data to and from the GPU
    manually. When the manual data movement is used, the `to` and `from` clauses
    will also need to reflect that, e.g. empty list.

    Also appy.target = 'numpy' will just disable the compilation, and run everything
    as is as a regular NumPy program. appy.target = 'numba' will replace the parallel
    for loop with `numba.prange` and import numba.
    '''
    c = np.empty_like(a)
    a, b, c = appy.to_gpu(a, b, c)
    #pragma parallel for simd to() from()
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    c = appy.from_gpu(c)
    return c


for f in [f6, f7]:
    src = inspect.getsource(f)
    newcode = appy.compile_from_src(src, dump_final_appy=1)
    print(newcode)