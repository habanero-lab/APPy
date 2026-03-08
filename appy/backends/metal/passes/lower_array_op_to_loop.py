import ast
from astpass.passes import vector_op_to_loop

class AddSIMDAnnotation(ast.NodeTransformer):
    def visit_For(self, node):
        if hasattr(node, "_simd_okay"):
            node.pragma = {
                "simd": True
            }

        return self.generic_visit(node)

def transform(tree, rt_vals):
    tree = vector_op_to_loop.transform(tree, rt_vals)
    tree = AddSIMDAnnotation().visit(tree)
    return tree
