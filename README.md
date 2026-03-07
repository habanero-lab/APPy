<!-- [![Documentation Status](https://readthedocs.org/projects/appy/badge/?version=latest)](https://appy.readthedocs.io/en/latest/?badge=latest) -->

# APPy

**APPy (Annotated Parallelism for Python)** enables Python `for` loops to run efficiently on GPUs with minimal code changes.

Instead of rewriting code using GPU frameworks, users simply annotate ordinary Python loops. The APPy compiler then automatically generates GPU kernels and manages data movement between the CPU and GPU.

Below is a simple example.

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

# Execute the function
increment_by_1(a)

# Inspect the result
print(a)  # [ 1  2  3  4  5  6  7  8  9 10]
```

Under the hood, APPy compiles the annotated loop into a GPU device function and launches it as a GPU kernel. The compiler also automatically generates code to:

* transfer input data from CPU memory to GPU memory before kernel execution
* execute the generated kernel on the GPU
* copy results back to the CPU after execution

This allows programmers to benefit from GPU acceleration while writing code that remains close to standard Python.


# Installation

To install the core APPy compiler:

```bash
pip install -e .
```

This installs the APPy code generator itself.

To execute the generated GPU code, you also need to install a backend runtime such as PyCUDA or Triton, depending on the platform.

For example, to install the Triton backend:

```bash
pip install -e .[triton]
```


# Quick Start

Additional examples are available in the `examples` directory.

To run a simple vector addition example:

```bash
python examples/01-vec_add.py
```


# Key Ideas

APPy introduces several design choices that make GPU acceleration accessible to Python programmers.

### 1. Preserve the Sequential Programming Model

APPy keeps the original sequential loop structure.

* Programmers write standard Python loops
* Parallelism is expressed through lightweight annotations

This minimizes code changes compared to GPU frameworks that require rewriting kernels.


### 2. Automatic Host–Device Memory Management

APPy automatically manages data transfers between CPU and GPU memory.

By default, users do not need to manually move arrays between host and device memory.


### 3. Automatic Parallelism Mapping

The compiler analyzes the loop body to determine available data parallelism and maps the computation efficiently to GPU threads. As a result, the programmer only needs to focus on "maximizing the amount of parallelism".


# Data Scope

## Arrays

Array variables must be defined before entering a parallel region.

Their data may reside either in:

* **CPU memory** (e.g., NumPy arrays)
* **GPU memory** (e.g., PyTorch CUDA tensors)

For arrays located in CPU memory, APPy automatically:

1. copies data to the GPU before launching the kernel
2. executes the kernel
3. copies the results back to the CPU

For arrays already located on the GPU, APPy leaves them in place and does not perform additional transfers.


## Scalar Variables

Scalar variables can be defined either outside or inside the parallel region.

If a scalar variable is defined **outside** the parallel region and used inside it, the value is passed to the kernel using **pass-by-value semantics**:

* the kernel receives the initial value
* updates inside the kernel are not visible outside by default

To make updates visible after the kernel finishes, the variable must be declared in the `shared` clause.

This instructs the compiler to:

1. copy the scalar value to GPU memory
2. allow it to be updated during kernel execution
3. copy the final value back to the CPU

Scalar variables defined **inside the parallel region** are treated as **local variables to each looop iteration**, which makes them safe for parallel execution.


# Parallel Reduction

The following example illustrates a parallel reduction:

```python
@appy.jit
def vector_sum(A):
    s = 0.0
    #pragma parallel for simd shared(s)
    for i in range(A.shape[0]):
        s += A[i]
```

The compiler automatically detects the reduction pattern and generates correct parallel code (for example, using atomic operations).

The clause `shared(s)` ensures that updates to `s` inside the kernel are visible after kernel execution.


# Array Expressions

In addition to simple loops, APPy also supports **1D array expressions inside parallel loops**.

Example:

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


# Supported Operations

APPy supports the following operations within a parallel region.

### Scalar operations (integers or floats)

* Arithmetic operations
* Mathematical functions (via the `math` package)
* Bitwise operations
* Logical operations
* Comparison operations

### Array operations

* Array indexing (load and store)

### Control flow

* Ternary operators


# Citation

If you use APPy in your research, please cite:

* [APPy: Annotated Parallelism for Python on GPUs](https://dl.acm.org/doi/10.1145/3640537.3641575)
