High-Level Programming Interface
================================

With APPy's high-level programming interface, all you need to do to parallelize a loop on GPUs 
is to replace ``range`` with ``appy.prange`` and annotate the loop with ``@appy.jit``. NumPy arrays
and scalar values can be directly used inside the ``prange`` region. Here's an example: 

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

In this example, ``a`` is a NumPy array and is used directly inside the ``prange`` region.
The compiler will handle the data movement between the CPU and the GPU automatically under the hood. 

.. note::
    Only code inside the ``prange`` region will be compiled and executed on the GPU. Other 
    code will be just executed by the Python interpreter on the CPU, i.e. no compilation happens.

Reductions can be parallelized as well with ``prange``. Here's an example:

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

APPy automatically detects reductions on scalar variables and generates atomic operations for them. 


Unsupported Features
====================

In general, it's fine to use common NumPy array operations, slicing, and indexing inside ``prange``. 
Control flows such as ``if`` structures are also supported, though using some of them could prevent automatic vectorization.

Unsupported Python language constructs inside ``prange`` include:

* Containers such as ``list`` and ``dict``.
* ``break`` and ``return`` statements (``continue`` is fine).
* ``try`` and ``except`` blocks.
* String operations.

Unsupported NumPy functions inside ``prange`` include:

* ``np.linalg``.
* ``np.random``.
* ``np.fft``.

Outside of the ``prange`` region, the code will simply be executed by the Python interpreter so there's no limitation.