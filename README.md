Annotated Parallelism for Python (APPy) is parallel programming model that allows you to parallelize your sequential loops or tensor expressions with annotations in the comments.


# Install

```bash
pip install -e .
```

# Quick Start

```bash
python tests/test_vec_add.py
```

# Notes

* APPy only works with pytorch device=cuda tensors for now. So if you have
numpy or cupy arrays, be sure to convert them to pytorch cuda tensors.
* Removing `@appy.jit` makes the function a normal Python function, which
could be helpful for debugging.
* The result of a reduction must not be used as a sub-expression.


# Element-Wise Operation
```python
@jit
def add(a, b, c, N, BLOCK=128):
    for i in range(0, N, BLOCK):  #pragma parallel
        c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]
```

or tensor operator based pragmas:
```python
@jit(auto_block=True)
def add(a, b, c, N, BLOCK=128):
    #pragma :N=>parallel
    c[:N] = a[:N] + b[:N]
```


# Grid Reduction

```python
@jit
def sum(a, b, N, BLOCK=256):
    b[0] = 0
    for i in range(0, N, BLOCK):  #pragma parallel reduction(b)
        b[0] += torch.sum(a[i:i+BLOCK])
```


# Stencils

`heat_3d` (from polybench)
```python
@appy.jit(dim_info={'A': ('M', 'N', 'K'), 'B': ('M', 'N', 'K')}, auto_block=True)
def kernel(TSTEPS, A, B):
    M, N, K = A.shape
    for t in range(1, TSTEPS):
        #pragma 1:M-1=>parallel 1:N-1=>parallel 1:K-1=>parallel
        B[1:-1, 1:-1,
            1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                                A[:-2, 1:-1, 1:-1]) + 0.125 *
                    (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                        A[1:-1, :-2, 1:-1]) + 0.125 *
                    (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                        A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])

        #pragma 1:M-1=>parallel 1:N-1=>parallel 1:K-1=>parallel
        A[1:-1, 1:-1,
            1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                                B[:-2, 1:-1, 1:-1]) + 0.125 *
                    (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                        B[1:-1, :-2, 1:-1]) + 0.125 *
                    (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                        B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])
    return A, B
```

# Sum of Matrix-Vector Multiplication
`gesummv` from polybench:

```python
@appy.jit(auto_block=True)
def kernel(alpha, beta, A, B, x):
    M, N = A.shape
    alpha, beta = float(alpha), float(beta)
    y = torch.empty([M], dtype=A.dtype, device=A.device)
    y1 = torch.empty_like(y)
    y2 = torch.empty_like(y)
    #pragma :M=>parallel,block(2) :N=>reduce(sum:y1)
    y1[:M] = torch.mv(alpha * A[:M, :N], x[:N])
    #pragma :M=>parallel,block(2) :N=>reduce(sum:y2)
    y2[:M] = torch.mv(beta * B[:M, :N], x[:N])
    #pragma :M=>parallel
    y[:M] = y1[:M] + y2[:M]
    return y
```