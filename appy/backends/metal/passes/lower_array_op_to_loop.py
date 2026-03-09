import ast
from astpass.passes import shape_analysis
from astpass.passes.vector_op_to_loop.convert_reduction_and_pointwise import ReductionAndPWExprToLoop

REDUCTION_OPS = {'np.sum', 'np.min', 'np.max', 'sum', 'min', 'max'}

class LowerArrayOpToLoop(ReductionAndPWExprToLoop):
    def gen_loop(self, node, low, up):
        result = super().gen_loop(node, low, up)
        if isinstance(result, tuple):
            # Reduction: (init, loop, reassign) — inspect original RHS for the op
            _, loop, _ = result
            reduce_op = ast.unparse(node.value.func)
            loop.pragma = {'simd': True, 'reduction': reduce_op}
        else:
            # Pointwise
            result.pragma = {'simd': True}
        return result

def transform(tree, rt_vals):
    shape_info = shape_analysis.analyze(tree, rt_vals)
    return LowerArrayOpToLoop(shape_info).visit(tree)
