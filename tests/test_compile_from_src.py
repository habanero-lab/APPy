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
        start, end = A_indptr[i], A_indptr[i+1]
        y[i] = 0.0
        #pragma simd
        for j in range(start, end):
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

for f in [f0, f1, f2, f3]:
    src = inspect.getsource(f)
    newcode = appy.compile_from_src(src)
    print(newcode)