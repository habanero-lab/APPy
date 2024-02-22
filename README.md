APPy (Annotated Parallelism for Python) enables users to parallelize generic Python loops and tensor expressions for execution on GPUs by simply adding compiler directives (annotations) to Python code. The foundation of APPy is described in [APPy: Annotated Parallelism for Python on GPUs](https://dl.acm.org/doi/10.1145/3640537.3641575). 

# Install

```bash
pip install -e .
```

# Quick Start

```bash
python tests/test_vec_add.py
```

# Loop-Oriented programming interface
## Parallelization
A loop can be parallelized by being annotated with `#pragma parallel for`, where the end of the loop acts as a synchronization point. Each loop iteration is said to be assigned to a *worker*, and the number of workers launched is always equal to the number of loop iterations, unless directive `#pragma sequential for` is used, which launches only one worker that executes all iterations, e.g. due to loop-carried dependences. Each worker is scheduled to a single vector processor, and executes its instructions sequentially. 
A parallel for-loop must be a for-range loop, and the number of loop iterations must be known at kernel launch time, i.e. no dynamic parallelism.

Tensors within the parallel region must already be a GPU tensor (data reside in the GPU memory), and currently `cupy` and `pytorch` are supported. Such libraries also provide APIs to create a GPU tensor from a NumPy array.

A vector addition example is shown below. Parallelize a for loop with APPy via `#pragma parallel for`. `#pragma ...` is a regular comment in Python, but will be parsed and treated as a directive by APPy.

```python
@appy.jit
def vector_add(A, B, C, N):
    #pragma parallel for
    for i in range(N):
        C[i] = A[i] + B[i]
```

## Vectorization
Although `#pragma parallel for` parallelizes a loop, maximum parallelism is achieved when the loop body is also vectorized, when applicable. APPy provides two high-level ways to achieve vectorization: 1) use tensor/array expressions (compiler generates a loop automatically); 2) annotate a loop with the `#pragma simd`, which divides the loop into smaller chunks.

Vector addition example.

```python
@appy.jit
def vector_add(A, B, C, N):
    #pragma parallel for simd
    for i in range(N):
        C[i] = A[i] + B[i]
```

SpMV example. 

```python
@appy.jit
def spmv(A_row, A_col, A_val, x, y, N):
    #pragma parallel for
    for i in range(N - 1):
        y[i] = 0.0
        #pragma simd
        for j in range(A_row[i], A_row[1+i]):            
            col = A_col[j]
            y[i] += A_val[j] * x[col]
```

A loop that is not applicable for parallelization may be vectorizable. One example is the `j` loop in the SpMV example, where it has dynamic loop bounds.

## Data Sharing
APPy does not require the programmer to manually specify whether each variable is private or shared. Instead, it enforces syntactical difference between array and non-array variables, and use simple rules to infer the scope of the variables. Array variables are always followed by a square bracket, such as `A[vi]`, and the rest are non-array variables. Array variables inside the parallel region are always considered shared (and their data reside in the global memory of the GPU). Non-array variables defined within the parallel region are considered private to each worker. Read-Only non-array variables are shared. APPy prohibits multiple reaching definitions from both inside and outside the parallel region of a non-array variable, which prevents writing into a shared non-array variable. To achieve such effects, the idiom is make the variable an array of size 1 (thus it has a global scope). 

## Parallel reduction
The only synchronization across workers (loop iterations) supported is atomically updating a memory location. This can be achieved by either using "assembly-level" programming of the abstract machine via instruction `appy.atomic_<op>`, or by annotating a statement with compiler directive `#pragma atomic`. Note that within a worker, no synchronization is necessary even if it works on multiple elements.

A parallel reduction example. 
```python
@appy.jit
def vector_sum(A, s, N):
    #pragma parallel for simd reduction
    for i in range(N):
        #pragma atomic
        s[0] += A[i]
```

# Tensor-Oriented programming interface 
In addition to loops, APPy also allows users to use tensor/array expressions and the tensors can have arbitrary size. This often results in more natural and succinct program compared to the loop oriented version. 

```python
@appy.jit(auto_simd=True)
def gesummv(alpha, beta, A, B, x, y, tmp, M, N):
    #pragma :M=>parallel :N=>reduction(sum:y)
    y[:M] = mv(alpha * A[:M, :N], x[:N])
    #pragma :M=>parallel :N=>reduction(sum:tmp)
    tmp[:M] = mv(beta * B[:M, :N], x[:N])
    #pragma :M=>parallel
    y[:M] += tmp[:M]

@appy.jit(auto_simd=True)
def jacobi_2d_one_iteration(A, B, M, N):
    #pragma 1:M-1=>parallel 1:N-1=>parallel
    B[1:M-1, 1:N-1] = 0.2 * (A[1:M-1, 1:N-1] + A[1:M-1, :N-2] + 
               A[1:M-1, 2:N] + A[2:M, 1:N-1] + A[0:M-2, 1:N-1])
    #pragma 1:M-1=>parallel 1:N-1=>parallel
    A[1:M-1, 1:N-1] = 0.2 * (B[1:M-1, 1:N-1] + B[1:M-1, :N-2] + 
               B[1:M-1, 2:N] + B[2:M, 1:N-1] + B[0:M-2, 1:N-1])
```

# Unsupported operations
Within an APPy parallel region, only elementary mathematical function and reduction functions are supported. Other operations including higher-level functions such as matrix inversion, shuffling functions such as sorting are not supported (will not compile).