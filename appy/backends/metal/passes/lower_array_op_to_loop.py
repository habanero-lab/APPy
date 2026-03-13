import ast
from astpass.passes import vector_op_to_loop


class AttachMetalSIMDPragmas(ast.NodeTransformer):
    def visit_For(self, node):
        self.generic_visit(node)
        if hasattr(node, '_simd_okay'):
            pragma = {'simd': True}
            if hasattr(node, '_reduction'):
                reduce_op, var_name = node._reduction
                pragma['reduction'] = reduce_op
                pragma['reduction_var'] = var_name
            node.pragma = pragma
        return node


def transform(tree, rt_vals):
    tree = vector_op_to_loop.transform(tree, rt_vals)
    return AttachMetalSIMDPragmas().visit(tree)
