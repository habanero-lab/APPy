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


for f in [f3]:
    src = inspect.getsource(f)
    newcode = appy.compile_from_src(src)
    print(newcode)