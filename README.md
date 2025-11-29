[![Documentation Status](https://readthedocs.org/projects/appy/badge/?version=latest)](https://appy.readthedocs.io/en/latest/?badge=latest)

APPy (Annotated Parallelism for Python) enables users to parallelize generic Python loops and tensor expressions for execution on GPUs by adding OpenMP-like compiler directives (annotations) to Python code. With APPy, parallelizing a Python for loop on the GPU can be as simple as adding a `#pragma parallel for` before the loop, like the following:

```python
#pragma parallel for
for i in range(N):
    C[i] = A[i] + B[i]
```

The APPy compiler will recognize the pragma, JIT-compile the loop to GPU code, and execute the loop on the GPU. A detailed description of APPy can be found in [APPy: Annotated Parallelism for Python on GPUs](https://dl.acm.org/doi/10.1145/3640537.3641575). This document provides a quick guide to get started. 

News: A web-based version of APPy (https://tongzhou80.github.io/appy-web/index.html) is now available to quickly try APPy online and view the generated code!

# Install

APPy tries to keep dependences minimal, to install the minimal version of APPy which only includes the code generator itself, run:

```bash
pip install -e .
```

In addition, if you want to be able to execute the generated GPU code, run:

```bash
pip install -e .[triton]
```

APPy currently has a [Triton](https://github.com/openai/triton/tree/main) backend, which requires `torch` and `triton` installed and a Linux platform with an NVIDIA GPU (Compute Capability 7.0+).


# Quick Start

```bash
python examples/01-vec_add.py
```

# Loop-Oriented programming interface
## Parallelization
A loop can be parallelized by being annotated with `#pragma parallel for`, where the end of the loop acts as a synchronization point. Each loop iteration is said to be assigned to a *worker*, and the number of workers launched is always equal to the number of loop iterations. Each worker is scheduled to a single vector processor, and executes its instructions sequentially. 
A parallel for-loop must be a for-range loop, and the number of loop iterations must be known at kernel launch time, i.e. no dynamic parallelism.

A vector addition example is shown below to parallelize a for loop with APPy via `#pragma parallel for`. `#pragma ...` is a regular comment in Python, but will be parsed and treated as a directive by APPy.

```python
@appy.jit
def vector_add(A, B, C, N):
    #pragma parallel for
    for i in range(N):
        C[i] = A[i] + B[i]
```

## APPy's Machine Model
A key design of APPy is that it assumes a simple abstract machine model, i.e. a multi-vector processor, instead of directly exposing the complex GPU architecture to the programmer. In this multi-vector processor, there are 2 layers of parallelism: 1) each vector processor is able to do vector processing (SIMD); 2) different vector processors run independently and simultaneously (MIMD). Pragma `#pragma parallel for` corresponds to the MIMD parallelism, which is also referred to as parallelization. The SIMD parallelism is referred to as vectorization, as described in more detail in the next section. Maximum parallelism is achieved with the loop is both parallelized and vectorized.

<p align="center">
  <img src="https://github.com/habanero-lab/APPy/assets/7697776/6425b55c-4148-4bac-9eae-e0fbab3cfa31" width=50% height=50%>
</p>


## Vectorization
Although `#pragma parallel for` parallelizes a loop, maximum parallelism is achieved when the loop body is also vectorized, when applicable. APPy provides two high-level ways to achieve vectorization: 1) use tensor/array expressions (compiler generates a loop automatically, though this feature is not included as of v0.3.0); 2) annotate a loop with the `#pragma simd`, which divides the loop into smaller chunks.

Vector addition example.

```python
@appy.jit
def vector_add(A, B, C):
    #pragma parallel for simd
    for i in range(A.shape[0]):
        C[i] = A[i] + B[i]
```

SpMV example. 

```python
@appy.jit
def spmv(A_row, A_col, A_val, x, y, N):
    #pragma parallel for
    for i in range(N - 1):
        yi = 0.0
        #pragma simd
        for j in range(A_row[i], A_row[1+i]):
            yi += A_val[j] * x[A_col[j]]
        y[i] = yi
```

A loop that is not applicable for parallelization may be vectorizable. One example is the `j` loop in the SpMV example, where it has dynamic loop bounds.

## Data Scope 
Array variables must already be defined before executing the parallel region, while their data can either reside in CPU memory or GPU memory. For CPU arrays, the compiler will automatically move data to the device before launching the kernel and move data back to the host after the kernel finishes. For GPU arrays, the compiler does not do move them, e.g. they stay where they are throughout the kernel. 

Scalar variables may be defined either outside the parallel region, or inside the parallel region. If defined outside and used inside, the variable has an "argument passing by value" semantic, where it gets the initial value from outside when the kernel is launched but any updates are only visible inside the kernel. To make the updates visible outside the kernel, the variable must be declared in the `shared` clause, which tells the compiler to copy the variable to GPU memory before the kernel is launched and copy it back after the kernel finishes. Scalar variables defined inside parallel region are considered local to each worker, e.g. can be safely parallelized.

## Parallel reduction
A parallel reduction example. 
```python
@appy.jit
def vector_sum(A):
    s = 0.0
    #pragma parallel for simd shared(s)
    for i in range(A.shape[0]):
        s += A[i]
```

The compiler automatically recognizes the parallel reduction pattern, and generates correct code for it, e.g. using atomic operations. Clause `shared(s)` makes the update to `s` inside the kernel visible outside the kernel, which essentially treats `s` as a single-element array.

# Tensor-Oriented programming interface 
Supporting tensor expressions inside parallel regions is future work! Currently only explicit loops are supported as of v0.3.0.

# Supported operations
APPy supports the following kinds of operations inside the parallel region:

On scalar integer or float values:

    Arithmetic operations
    Math functions (via the math package)
    Bitwise operations
    Logical operations
    Compare operations

On arrays of integers or floats:

    Array indexing (store or load)

Control flows:

    Ternary operators
