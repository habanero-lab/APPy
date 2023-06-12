BMP (Blocked Multi-Processing) is a programming model and a compiler for simplified parallel programming. It employs the *Single Program Multiple Blocked Data* paradigm where the outer loops are sequential for loops annotated with OpenMP-like pragmas, and inner statements operate on blocks of data. The programming model is like SPMD, where we launch multiple instances of the same program, but with each program working on a contiguous block of data. As a result, user can write sequential Python loops, debug in Python interpreter, and compile and execute them on GPUs and CPUs.


# Install

```bash
python setup.py develop
```

# Element-Wise Operation
```python
@slap.jit()
def add(a, b, c, BLOCK):
    for i in range(0, a.shape[0], BLOCK: parallel):  #pragma parallel
        c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]
```

With `auto-tile`:

```python
@slap.jit(auto_tile=True)
def add(a, b, c):
    for i in range(a.shape[0]):  #pragma parallel
        c[i] = a[i] + b[i]
```

# Grid Reduction

```python
@slap.jit()
def kernel(a, b, BLOCK):
    b[0] = 0
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel reduction(+:b)
        b[0] += torch.sum(a[i:i+BLOCK])
```

Or
```python
@slap.jit(auto_tile=True)
def kernel(a, b):
    b[0] = 0
    for i in range(a.shape[0]):  #pragma parallel reduction(+:b)
        b[0] += a[i]
```

# Indirect reduction

```python
@slap.jit()
def kernel(x, labels, centers):
    for i in range(x.shape[0]):  #pragma parallel reduction(+:centers) indirect index
        for j in range(0, x.shape[1], Bj):  #pragma parallel
            label = labels[i]
            centers[label,j:j+Bj] += x[i,j:j+Bj]
```

Or 
```python
@slap.jit(auto_tile=True)
def kernel(x, labels, centers):
    for i in range(x.shape[0]):  #pragma parallel reduction(+:centers) indirect index
        for j in range(x.shape[1]):  #pragma parallel
            label = labels[i]
            centers[label,j] += x[i,j]
```

# Matrix Multiplications

An blocked matrix multiplication implementation can be expressed as:
```python
@slap.jit()
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

@slap.jit()
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
@slap.jit()
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
@slap.jit()
def kernel(a_rowptrs, a_cols, a_vals, b, c, BLOCK):
    for i in range(a.shape[0]):  #pragma parallel indirect index
        for j in range(0, b.shape[1], BLOCK):  #pragma parallel 
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                c[i,j:j+BLOCK] += a_ik * b[ks,j:j+BLOCK]
```

Or
```python
@slap.jit(auto_tile=True)
def kernel(a_rowptrs, a_cols, a_vals, b, c):
    for i in range(a.shape[0]):  #pragma parallel indirect index
        for j in range(b.shape[1]):  #pragma parallel 
            for ki in range(a_rowptrs[i], a_rowptrs[i+1]):
                a_ik = a_vals[ki]
                ks = a_cols[ki]
                c[i,j] += a_ik * b[ks,j]
```