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

or 
```python
@jit
def kernel(a, b, N, BLOCK=256):
    #pragma par_dim(0:N:BLOCK)
    b[0] = torch.sum(a[:N])
```

# Dividing by Sum
```python
@jit
def sum(a, b, N, BLOCK=256):
    b[0] = 0
    for i in range(0, N, BLOCK):  #pragma parallel reduction(b)
        b[0] += torch.sum(a[i:i+BLOCK])

    for i in range(0, N, BLOCK):  #pragma parallel
    	a[i:i+BLOCK] /= b[0]
```

or 
```python
@jit
def kernel(a, b, N, BLOCK=256):
    #pragma par_dim(0:N:BLOCK)
    a[:N] = a[:N] / torch.sum(a[:N])
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
def kernel(x, labels, centers, Bj=128):
    #pragma parallel reduction(centers)
    for i in range(x.shape[0]):
    	#pragma par_dim(0:x.shape[1]:Bj)
    	centers[labels[i],j] += x[i,j]
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

Or

```python
@jit
def matmul(a, b, c, M, N, K, Bi, Bj, Bk):
    #pragma par_dim(:M:BM, :N:BN) seq_dim(:K:BK)
    c[:M, :N] = a[:M, :K] @ b[:K, :N]
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

Or

```python
@jit
def matmul(a, b, c, M, N, K, Bi, Bj, Bk):
    #pragma par_dim(:M:BM, :N:BN, :K:BK)
    c[:M, :N] = a[:M, :K] @ b[:K, :N]
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

Or

```python
@jit
def matmul(a, b, c, Z, M, N, K, Bi, Bj, Bk):
    #pragma par_dim(:Z, :M:BM, :N:BN) seq_dim(:K:BK)
    c[:Z, :M, :N] = a[:Z, :M, :K] @ b[:Z, :K, :N]
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