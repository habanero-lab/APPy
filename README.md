BMP (Blocked Multi-Processing) is a programming model and a compiler for simplified parallel programming. It employs the *Single Program Multiple Blocked Data* paradigm where the outer loops are sequential for loops annotated with OpenMP-like pragmas, and inner statements operate on blocks of data. The programming model is like SPMD, where we launch multiple instances of the same program, but with each program working on a contiguous block of data. As a result, user can write sequential Python loops, debug in Python interpreter, and compile and execute them on GPUs and CPUs.


# Install

```bash
python setup.py develop
```

# Element-Wise Operation
```python
@slap.jit(tune=['BLOCK'])
def add(a, b, c, BLOCK):
    for i in range(0, a.shape[0], BLOCK: parallel):  #pragma parallel
        c[i:i+BLOCK] = a[i:i+BLOCK] + b[i:i+BLOCK]
```

# Grid Reduction

```python
<<<<<<< HEAD
@slap.jit(tune=['BLOCK'])
def kernel(a, b, BLOCK):
=======
@bmp.jit(tune=['BLOCK'])
def kernel(a, b, BLOCK: parallel):
>>>>>>> 9a9b05462f966fec1d8eeb6ce5d37fde73a76a5d
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel reduction(+:b)
        s = torch.sum(a[i:i+BLOCK])
        b[0] += s 
```

# Matrix Multiplications

An blocked matrix multiplication implementation can be expressed as:
```python
@slap.jit(tune=['Bi', 'Bj', 'Bk'])
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

@slap.jit(tune=['Bi', 'Bj', 'Bk'])
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
@slap.jit(tune=['Bi', 'Bj', 'Bk'])
def batched_matmul(a, b, c, Bi, Bj, Bk):
    for z in range(a.shape[0]):  #pragma parallel
        for i in range(0, a.shape[1], Bi):  #pragma parallel
            for j in range(0, b.shape[-1], Bj):  #pragma parallel
                c_block = torch.zeros([Bi, Bj], device=a.device, dtype=a.dtype)
                for k in range(0, a.shape[-1], Bk):
                    c_block[:, :] += a[z, i:i+Bi, k:k+Bk] @ b[z, k:k+Bk, j:j+Bj]
                c[z, i:i+Bi, j:j+Bj] = c_block
```

