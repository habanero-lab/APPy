<!-- [![Documentation Status](https://readthedocs.org/projects/appy/badge/?version=latest)](https://appy.readthedocs.io/en/latest/?badge=latest) -->

APPy (Annotated Parallelism for Python) makes it easy to run Python for loops on GPUs. Here's a simple example to get started:

```python
import numpy as np
import appy

@appy.jit
def increment_by_1(a):
    #pragma parallel for simd
    for i in range(a.shape[0]):
        a[i] += 1

# Create the input array
a = np.arange(10)

# Apply the increment_by_1 function
increment_by_1(a)

# Inspect the array after
print(a)  # Should print [ 1  2  3  4  5  6  7  8  9 10]
```

Under the hood, the compiler will generate a GPU device function that performs the increment operation, and pass input array `a` to the GPU kernel to execute on the GPU. In addition, the compiler also emits code to move data from CPU main memory to GPU memory before the kernel invocation, and move the data back to CPU after.

# Install

For a minimal installation which only includes the APPy code generator itself, run:

```bash
pip install -e .
```

To run the APPy generated code, you'd also need to install the backend packages, e.g. PyCUDA, Triton etc, depending on the platform. For example, to install the Triton backend, you could run 

```bash
pip install -e .[triton]
```


# Quick Start

The `examples` directory contains some more examples to get started:

```bash
python examples/01-vec_add.py
```

# Key Innovations
APPy offers three key innovations which makes it intuitive for Python programmers who'd like to accelerate loops on GPUs:

* Sequential loop structure is maintained, which brings minimal code change
    - User intents are communicated via very simple annotations
* Automatic memory management between host and device
    - No need to manage host-device data movement by default
* Automatic parallelism mapping depending on loop body code
    - Compiler detects the degree of data parallelism in the loop body and maps efficiently to hardware threads


# Data Scope 
Array variables must already be defined before executing the parallel region, while their data can either reside in CPU memory or GPU memory. For CPU arrays, e.g. NumPy arrays, the compiler will automatically move data to the device before launching the kernel and move data back to the host after the kernel finishes. For GPU arrays, e.g. PyTorch CUDA tensors, the compiler does not do move them, e.g. they stay where they are throughout the kernel. 

Scalar variables may be defined either outside the parallel region, or inside the parallel region. If defined outside and used inside, the variable has an "argument passing by value" semantic, where it gets the initial value from outside when the kernel is launched but any updates are only visible inside the kernel. To make the updates visible outside the kernel, the variable must be declared in the `shared` clause, which tells the compiler to copy the variable to the GPU memory where it can be updated and copy it back after the kernel finishes. Scalar variables defined inside parallel region are considered local to each worker, e.g. can be safely parallelized.

# Parallel reduction
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

Besides pure loops, 1D array expressions can also be used inside a parallel for loop, for example:

```python
@appy.jit
def syrk(alpha, beta, C, A):
    #pragma parallel for
    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]
    return C
```

# Supported operations
APPy supports the following kinds of operations inside the parallel region:

On scalar integer or float values or a 1D slice of an array:

    Arithmetic operations
    Math functions (via the math package)
    Bitwise operations
    Logical operations
    Compare operations

On arrays of integers or floats:

    Array indexing (store or load)

Control flows:

    Ternary operators

# Citations
We'll be grateful if you could cite the following publications for APPy:

- [APPy: Annotated Parallelism for Python on GPUs](https://dl.acm.org/doi/10.1145/3640537.3641575)