import ast
from astpass.passes import vector_op_to_loop
from astpass.passes import normalize_ranges

class AddSIMDAnnotation(ast.NodeTransformer):
    def visit_For(self, node):
        if hasattr(node, "_simd_okay"):
            node.pragma = {
                "simd": True
            }

        if hasattr(node, "_reduction"):
            reduce_op, var = node._reduction
            node.pragma["reduction"] = f"{'+' if reduce_op == 'sum' else reduce_op}:{var}"

        return self.generic_visit(node)

def transform(tree, rt_vals):
    # A quick fix to add numpy into the scope
    import numpy as np
    tree = vector_op_to_loop.transform(tree, rt_vals | {'np': np})
    tree = normalize_ranges.transform(tree)
    tree = AddSIMDAnnotation().visit(tree)
    return tree