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

This rule works well for detecting reductions on scalar variables (it's sound for commonly used reduction patterns). However, it may easily have false positives
for array variables.

.. code-block:: python
    
    @jit
    def add_one(a):
         for i in prange(a.shape[0]):
              a[i] += 1

In the ``add_one`` example above, all assignments to ``a[i]`` are in the form of ``a[i] = a[i] + 1``. 
Therefore the previous rule will detect ``a[i]`` as a reduction pattern. However, ``a[i] = a[i] + 1`` is not a reduction pattern here, as it performs an element-wise addition on array elements.

To reduce false positives, we can check the indices of the array variable in the assignment. 
The previous rule is extended to distinguish reduction with regard to which loop is involved:

* For loop ``loop_i`` with index variable ``i``, an array store ``a[i0, i1, ...]`` is considered a reduction pattern 
  if all the assignments to it has the same reduction operator, and ``i`` does not appear in the array store indices ``i0, i1, ...``.

In the ``add_one`` example, the loop index variables ``i`` does appear in the array store ``a[i]``, thus ``a[i] = a[i] + 1`` is not a reduction with regard to loop ``i``.
On the other hand, ``a[j] += b[i,j]`` in the following code is a reduction with regard to loop ``i``, but not a reduction with regard to loop ``j``:

.. code-block:: python
    
    @jit
    def foo(a):
         for i in prange(a.shape[0]):
                for j in range(a.shape[1]):
                    a[j] += b[i,j]


Similarly, for the code example below with ``a[i] += b[i,j]``, ``a[i]`` will be detected as a reduction with regard to loop ``j``, but not to loop ``i``:
 
.. code-block:: python
    
    @jit
    def foo(a, b):
         for i in prange(a.shape[0]):
                for j in range(a.shape[1]):
                    a[i] += b[i,j]

.. code-block:: python

    class DetectReduction(ast.NodeVisitor):
        def __init__(self):
            self.loop_stack = []
            self.candidates = {}
    
        def visit_For(self, node):
            index = node.target.id
            self.loop_stack.append(index)
            # Visit sub-loops
            self.generic_visit(node)

            # Need an assignment visitor here
            self.visit_Assign(node)

            self.loop_stack.pop()

        def get_reduction_op(self, node):
            pass

        def visit_Assign(self, node):
            target = node.targets[0]
            reduce_op = self.get_reduction_op(node)
            if reduce_op is not None:
                if is_ast_name(target):
                    # If is scalar, use the previous rule
                    self.candidates.set_default(target.id, set()).add(reduce_op)
                elif is_array_store(target):
                    # If is array, use the extended rule
                    indices = get_array_store_indices(node)
                    if indices == self.loop_stack:
                        # Not a reduction candidate
                        pass
                    else:
                        self.candidates.setdefault(unparse(target), set()).add(reduce_op)
                    



Low-Level Programming Interface
-------------------------------

APPy utilizes Triton for its backend for low-level programming interface. 