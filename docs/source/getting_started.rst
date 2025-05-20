Getting Started with APPy
=========================

This guide walks you through how to install and use APPy to parallelize your Python loops on GPUs.

Installation
------------

APPy is available on PyPI as `appyc` and can be installed using pip:

.. code-block:: bash

    pip install appyc


Basic Example
-------------

The easiest way to parallelize a Python loop with APPy is to replace ``range`` with ``appy.prange``
and annotate the loop with ``@appy.jit``:

.. code-block:: python

   from appy import jit, prange

   @jit
   def add_vectors(a, b, c):
       for i in prange(len(a)):
           c[i] = a[i] + b[i]

   # Call the function with NumPy arrays
   # APPy will compile this to GPU code automatically


When can ``appy.prange`` be used?
---------------------------------

``prange`` may be used if the loop does not have any cross-iteration dependencies, except for reductions which can actually be parallelized.

An example of a cross-iteration dependency is:

.. code-block:: python

   def add_vectors(a, N):
       for i in range(N):
           a[i+1] = a[i]

In this code example, every loop iteration depends on the previous loop iteration, so the loop cannot be parallelized (``prange`` cannot be used).

Reduction is a special case of cross-iteration dependency, but can be parallelized:

.. code-block:: python

   @jit
   def add_vectors(a, N):
       sum = 0
       for i in prange(N):
           sum += a[i]
       return sum

More examples are available in the tutorials section.

