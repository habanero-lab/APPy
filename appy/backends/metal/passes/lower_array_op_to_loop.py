import ast
from .shape_analysis import analyze
from astpass.passes.vector_op_to_loop import ExtractReductionSubexprs, ReductionAndPWExprToLoop


class MyReductionAndPWExprToLoop(ReductionAndPWExprToLoop):
    def visit_Assign(self, node):
        # Special handling for "xx = appy.buffer(size, dtype)"
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "appy.buffer":
            return node
        else:
            return super().visit_Assign(node)

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
    tree = ExtractReductionSubexprs().visit(tree)
    shape_info = analyze(tree, rt_vals)
    tree = MyReductionAndPWExprToLoop(shape_info).visit(tree)
    return AttachMetalSIMDPragmas().visit(tree)
