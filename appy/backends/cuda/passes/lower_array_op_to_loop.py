import ast
from astpass.passes import shape_analysis
from astpass.passes.vector_op_to_loop.convert_reduction_and_pointwise import ReductionAndPWExprToLoop

REDUCTION_OP_MAP = {
    'np.sum': 'sum', 'sum': 'sum',
    'np.min': 'min', 'min': 'min',
    'np.max': 'max', 'max': 'max',
}


class LowerArrayOpToLoop(ReductionAndPWExprToLoop):
    def gen_loop(self, node, low, up):
        result = super().gen_loop(node, low, up)
        if isinstance(result, tuple):
            init_stmt, loop, _ = result
            var_name = init_stmt.targets[0].id
            canonical_op = REDUCTION_OP_MAP[ast.unparse(node.value.func)]
            loop.pragma = {'simd': True, 'reduction': canonical_op, 'reduction_var': var_name}
        else:
            result.pragma = {'simd': True}
        return result


def transform(tree, rt_vals):
    shape_info = shape_analysis.analyze(tree, rt_vals)
    return LowerArrayOpToLoop(shape_info).visit(tree)
