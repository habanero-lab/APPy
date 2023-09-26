Annotated Parallelism for Python (APPy) is parallel programming model that allows you to parallelize your sequential loops or tensor expressions with annotations in the comments.

APPy supports two programming models:

- A block-oriented (Vanilla) programming model, where the user uses explicit loops and annotate the loops with OpenMP/OpenACC like pragmas. Unlike OpenMP where a loop iteration works on typically only one data element, in APPy, it's recommended to make each loop iteration work on a block of data at a time, e.g. 1 to 2048 elements. Depending on the specific device, 256 is a good starting point to explore various block sizes. The best block size is hardware-specific and will need to be tuned.
- A tensor-oriented programming model, where the user is allowed to directly annotate tensor expressions. In the vanilla programming model, the user is restricted to work with a small block of data at a time, which can be cumbersome. The tensor oriented model allows users to directly work with tensors with arbitrary size and dimensions. 

Expressing vector addition using the vanilla model is as follows:

```python
@jit
def kernel_block_oriented(a, b, c, N, BLOCK=128):
    #pragma parallel
    for i in range(0, N, BLOCK):  
        vi = appy.vidx(i, BLOCK, bound=N)
        c[vi] = a[vi] + b[vi]
```
where `#pragma parallel` is equivalent to `#pragma parallel for` in OpenMP or OpenACC. Note that how each loop iteration works on `BLOCK` elements. Two key pieces are
* Have a parallel loop (can be nested)
* Work with 1-2048 elements per loop iteration


Or use the tensor based pragmas:
```python
@appy.jit(auto_block=True)
def kernel_tensor_oriented(a, b, c, N, BLOCK=128):
    #pragma :N=>parallel
    c[:N] = a[:N] + b[:N]
```
Each tensor expression must have all dimensions named explicitly using slices, e.g. `:N`. And in the annotation, each dimension must appear and be specified a set of possible properties. In the example above, there's only one dimension (`:N`), and its property is `parallel`, specified using syntax `dimension=>property1,property2,...`.

The tensor oriented programming model is higher level than the block oriented model, and no longer requires the user to explicitly block the operation. Instead, the user can enable option `auto_block=True` to let the compiler automatically block the tensor operation. But the user do need to specify whether a dimension is parallel or not, at minimum. On top of that, the compiler also performs more automatic optimizations with the tensor oriented model.

# Install

```bash
pip install -e .
```

# Quick Start

```bash
python tests/test_vec_add.py
```

# Notes

* APPy requires the tensor arguments to be pytorch cuda tensors for now. So if you have
numpy or cupy arrays, be sure to convert them to pytorch cuda tensors.
* APPy only supports compiling basic element-wise and reduction operations. Higher level functions like `numpy.linalg.inv` are not supported.
* Removing `@appy.jit` makes the function a normal Python function, which
could be helpful for debugging.
* The result of a reduction must not be used as a sub-expression.


More examples.

# Grid Reduction

```python
@appy.jit
def kernel(a, b, N, BLOCK=512):
    #pragma parallel
    for i in range(0, N, BLOCK):  
        vi = appy.vidx(i, BLOCK, bound=N)
        #pragma atomic
        b[0] += torch.sum(a[vi])
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