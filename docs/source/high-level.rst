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
