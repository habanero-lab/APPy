Getting Started with APPy
=========================

This guide walks you through how to install and use APPy to parallelize your Python loops on GPUs.

Installation
------------

APPy is available on PyPI as *appyc* and can be installed using pip:

.. code-block:: bash

    pip install appyc

Or you can install the latest development version from `source <https://github.com/habanero-lab/APPy>`_:

.. code-block:: bash

    git clone https://github.com/habanero-lab/APPy.git
    cd APPy
    pip install -e .

APPy is designed to have minimal dependencies and the ``appyc`` package only includes the code generator itself. 
To use the ``triton`` backend, you will also need to have ``torch`` and ``triton`` (part of the ``torch`` package) installed.

.. code-block:: bash

    pip install torch  

Supported platforms
-------------------
APPy currently supports Python 3.9+ on Linux platforms with a CUDA-enabled GPU (Compute Capability 8.0 or higher).


Basic example
-------------

The easiest way to parallelize a Python/NumPy loop with APPy is to replace ``range`` with ``appy.prange``
and annotate the loop with ``@appy.jit``:

.. code-block:: python

    import numpy as np
    from appy import jit, prange

    @jit
    def add_one(a):
         for i in prange(a.shape[0]):
              a[i] += 1

    a = np.zeros(10)
    add_one(a)

    # a is now [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

Reductions can be parallelized as well:

.. code-block:: python

    @jit
    def sum_vector(a, N):
        sum = 0
        for i in prange(N):
            sum += a[i]
        return sum

    a = np.ones(10)
    sum_vector(a, 10)

    # sum is now 10

APPy automatically detects reductions and make them work properly in parallel.


Will APPy work for my code?
---------------------------

APPy only supports the following operations in the parallel loop region, which should be sufficient to express a wide range of applications already.

On scalar integer or float values:

- Arithmetic operations
- Math functions
- Bitwise operations
- Logical operations
- Compare operations

On arrays of integers or floats:

    Array indexing (store or load)

Structurally, control flows except ``break`` and ``return`` can be used.

In general, APPy's usage scenarios are similar to `numba.prange <https://numba.readthedocs.io/en/stable/user/parallel.html>`_ which parallelizes Python loops on CPUs.

When can ``appy.prange`` be used?
---------------------------------

``prange`` may be used if the loop does not have any cross-iteration dependencies, except for reductions which can actually be parallelized.

An example of a cross-iteration dependency is:

.. code-block:: python

   def dependence_example(a, N):
       for i in range(N-1):
           a[i+1] = a[i]

In this code example, every loop iteration depends on the previous loop iteration, so the loop cannot be parallelized (``prange`` cannot be used).

Reduction is a special case of cross-iteration dependency that can be parallelized due to reduction operations being commutative:

.. code-block:: python

   @jit
   def sum_vector(a, N):
       sum = 0
       for i in prange(N):
           sum += a[i]
       return sum

More examples are available in :doc:`high-level` and :doc:`low-level`. 
APPy supports both a high-level and a low-level programming interface.
The high-level interface is easy to use - parallelizing a Python loop on GPUs 
is as simple as replacing ``range`` with ``appy.prange`` while
the low-level interface is more flexible and allows for more control over the generated code via pragmas.