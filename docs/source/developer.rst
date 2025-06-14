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

In this rule, control flows are fine - even if there are control flows, it's still a reduction pattern. However, 
we note that this approach will not properly detect reductions on array variables. 

.. code-block:: python
    
    @jit
    def add_one(a):
         for i in prange(a.shape[0]):
              a[i] += 1

In the ``add_one`` example above, all assignments to ``a[i]`` are in the form of ``a[i] = a[i] + 1``. 
This rule will detect ``a[i]`` as a reduction pattern. However, this loop performs an element-wise addition
on array elements, which is not a reduction pattern. 

Therefore, we extend the previous rule to the following detection algorithm:

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