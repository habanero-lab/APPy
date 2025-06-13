Release Notes
=============

This page summarizes major updates for each release of APPy.

APPy 0.2.x
----------

0.2.1
-----
* Supported prange loops with automatic parallelization and vectorization.
* Supported Jun's reduction detection approach on scalar variables.
* Supported automatic data movement for both scalar and array variables. The compiler is able to copy back into the original CPU memory space of an array after the kernel execution.


0.2.0
-----

* Added basic framework for the new high-level programming model. 
* Supported compiler option ``add_entry_exit_data_transfer``, i.e. manual data movement via ``appy.to_device`` etc is no longer necessary in many cases.

APPy 0.1.x
----------

APPy 0.1.x is based on our previous work on `Annotated Parallelism for Python on GPUs <https://dl.acm.org/doi/10.1145/3640537.3641575>`_.
It introduces 

* A multi- vector processor based abstract machine model.
* A set of pragma annotations for Python loops and tensor expressions.
* A Triton-based backend for JIT-compiled code generation.
