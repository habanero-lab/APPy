Developer Guide
===============

This guide explains how APPy compiler works under the hood.

High-Level Programming Interface
--------------------------------

APPy's high-level programming interface is a wrapper around the low-level programming interface where
the compiler performs three key automatic transformations:

1. **Data Movement**: APPy generates code that moves data between the CPU and the GPU automatically around the ``prange`` region.

2. **Reduction Detection**: APPy automatically detects reductions and add corresponding pragma annotations.

3. **Inner Loop Vectorization**: APPy automatically adds the `simd` pragma to innermost loops that can be vectorized.


Reduction Detection
-------------------

To detect reductions on scalar variable ``x``, APPy uses the following rules:

* Check all assignments to ``x``, ``x`` is a reduction pattern if and only if all the assignments to it has the same reduction operator. For example, all assignments to ``x`` are in the form of ``x = x + y``.

In this rule, control flow is fine - even if there are control flows, it's still a reduction pattern. 

.. note::
    This rule will not detect reductions on array variables. Users will need to manually annotate the reduction statements with ``#pragma atomic``.


Low-Level Programming Interface
-------------------------------

APPy utilizes Triton for its backend for low-level programming interface. 