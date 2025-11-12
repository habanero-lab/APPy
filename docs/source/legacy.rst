Legacy Programming Interface
===============================

APPy is first introduced in the 2024 CC paper which has a pragma-based programming interface. The two most used pragmas are ``#pragma parallel for`` and ``#pragma simd``.
These two have basically been replaced by ``appy.prange`` (``appy.range(parallel=True)``) and ``appy.range(simd=True)``. 
The compiler will also automatically detect vectorizable pattern even if option ``simd=True`` is not specified.
All other pragmas are not being supported anymore and are removed from the user programming interface.

The legacy ``#pragma parallel for`` and ``#pragma simd`` are deprecated and will also be removed in a future release.