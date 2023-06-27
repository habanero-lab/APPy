Annotated Parallelism for Python Loops (APPyL) is parallel programming model that allows you to parallelize your sequential loops with annotations in the comments.


# Install

```bash
python setup.py develop
```

# Clauses
A clause starts with `#pragma`:
- `parallel`
  - Indicate different iterations of the loop can run in parallel in SPMD style.
- `block(BLOCKSIZE)`
  - To block the loop, for SIMD parallelism or data reuse.
  - With an optional blocksize, which defaults to 128 if not specified.
  - Two conditions to block a loop
    - Adjacent loop iterations apply the same operation (to different data)
    - Adjacent loop iterations reuse the same data
  - Supported with contraints
    - A with annotated with `block` cannot be annotated with `reduction` at the same time.
    - No other loops are blocked, automatically via pragma or mannually.
- `reduction(var)`
  - Indicate a reduction pattern on variable `var`. 

# Element-Wise Operation
```python
@jit
def add(a, b, c, BLOCK=128):
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel
        c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]
```

In addition to manually block an index, one can also specify which loop to block in the pragma, with a block size parameter. 

```python
@jit
def add(a, b, c, BLOCK=128):
    for i in range(a.shape[0]):  #pragma parallel block(BLOCK)
        c[i] = a[i] + b[i]
```

# Grid Reduction

```python
@jit
def kernel(a, b, BLOCK):
    b[0] = 0
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel reduction(b)
        b[0] += torch.sum(a[i:i+BLOCK])
```

# Indirect Reduction

```python
@jit
def kernel(x, labels, centers, Bj=128):
    for i in range(x.shape[0]):  #pragma parallel reduction(+:centers) 
        for j in range(0, x.shape[1], Bj):  #pragma parallel
            label = labels[i]
            centers[label,j:j+Bj] += x[i,j:j+Bj]
```

Or 
```python
@jit
def kernel(x, labels, centers, Bj):
    for i in range(x.shape[0]):  #pragma parallel reduction(centers)
        for j in range(x.shape[1]):  #pragma parallel block(Bj)
            label = labels[i]
            centers[label,j] += x[i,j]
```

# Matrix Multiplications

An blocked matrix multiplication implementation can be expressed as:
```python
@jit
def matmul(a, b, c, Bi, Bj, Bk):
    for i in range(0, a.shape[0], Bi):  #pragma parallel
        for j in range(0, b.shape[-1], Bj):  #pragma parallel
            c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
            for k in range(0, a.shape[-1], Bk):
                c_block[:, :] += a[i:i+Bi, k:k+Bk] @ b[k:k+Bk, j:j+Bj]
            c[i:i+Bi, j:j+Bj] = c_block
```

When `k` dimension is relatively large, the following kernel reduces `k` in parallel:
```python
import slap

@jit
def matmul(a, b, c, Bi, Bj, Bk):
    # The first kernel launch
    for i in range(0, a.shape[0], Bi):  #pragma parallel
    	for j in range(0, b.shape[-1], Bj):  #pragma parallel
	        c[i:i+Bi, j:j+Bj] = 0

    # This will be the second kernel launch
    for i in range(0, a.shape[0], Bi):  #pragma parallel
    	for j in range(0, b.shape[-1], Bj):  #pragma parallel
	        for k in range(0, a.shape[-1], Bk):  #pragma parallel reduction(+:c)
	    	    c[i:i+Bi, j:j+Bj] += a[i:i+Bi, k:k+Bk] @ b[k:k+Bk, j:j+Bj]
```


Batched matmul is as easy as adding one extra outer loop and some minor changes:

```python
@jit
def batched_matmul(a, b, c, Bi, Bj, Bk):
    for z in range(a.shape[0]):  #pragma parallel
        for i in range(0, a.shape[1], Bi):  #pragma parallel
            for j in range(0, b.shape[-1], Bj):  #pragma parallel
                c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
                for k in range(0, a.shape[-1], Bk):
                    c_block[:, :] += a[z, i:i+Bi, k:k+Bk] @ b[z, k:k+Bk, j:j+Bj]
                c[z, i:i+Bi, j:j+Bj] = c_block
```

# Sparse-Dense Matrix Multiplication
```python
@jit
def kernel(a_rowptrs, a_cols, a_vals, b, c, BLOCK):
    for i in range(a.shape[0]):  #pragma parallel
        for j in range(0, b.shape[1], BLOCK):  #pragma parallel 
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                c[i,j:j+BLOCK] += a_ik * b[ks,j:j+BLOCK]
```

Or
```python
@jit
def kernel(a_rowptrs, a_cols, a_vals, b, c, BLOCK):
    for i in range(a.shape[0]):  #pragma parallel
        for j in range(b.shape[1]):  #pragma parallel block(BLOCK)
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                c[i,j] += a_ik * b[ks,j]
```