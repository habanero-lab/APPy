BMP (Blocked Multi-Processing) is a programming model and a compiler for simplified GPU programming. It employs the *single program multiple blocked data* paradigm where the outer loops are sequential for loops annotated with OpenMP-like pragmas, and inner statements operate on blocks of data.

# Install

```bash
python setup.py develop
```

# Element-Wise Function


# Matrix Multiplications

An blocked matrix multiplication implementation can be expressed as:
```python
import bmp

@bmp.jit(tune=['Bi', 'Bj', 'Bk'])
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
import bmp

@bmp.jit(tune=['Bi', 'Bj', 'Bk'])
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
@bmp.jit(tune=['Bi', 'Bj', 'Bk'])
def batched_matmul(a, b, c, Bi, Bj, Bk):
    for z in range(a.shape[0]):  #pragma parallel
        for i in range(0, a.shape[1], Bi):  # pragma parallel
    	    for j in range(0, b.shape[-1], Bj):  # pragma parallel
	        c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
	    	for k in range(0, a.shape[-1], Bk):
	    	    c_block[:, :] += a[z, i:i+Bi, k:k+Bk] @ b[z, k:k+Bk, j:j+Bj]
	    c[z, i:i+Bi, j:j+Bj] = c_block
```

