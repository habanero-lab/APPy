import ast
from ..constants import SIMD_WIDTH

class RewriteSIMDLoops(ast.NodeTransformer):
    """
    Rewrites SIMD-annotated for loops into strided loops over the lane index.

    A loop like:
        for j in range(low, up):  # pragma simd
    becomes:
        for j in range(lane, up, SIMD_WIDTH):

    This distributes the loop iterations across SIMD_WIDTH threads, where
    each thread handles iterations starting at its lane index with stride
    SIMD_WIDTH.
    """
    def visit_For(self, node):
        self.generic_visit(node)
        if not (hasattr(node, 'pragma') and node.pragma.get('simd')):
            return node

        range_args = node.iter.args
        assert len(range_args) in (1, 2), \
            f"SIMD for-range loop must have 1 or 2 arguments, got: {ast.unparse(node.iter)}"

        if len(range_args) == 1:
            low_node = None
            up = range_args[0]
        else:
            low_node = range_args[0]
            up = range_args[1]

        # new start = lane (if low==0) or lane + low
        if low_node is None:
            new_start = ast.Name(id='lane', ctx=ast.Load())
        else:
            new_start = ast.BinOp(
                left=ast.Name(id='lane', ctx=ast.Load()),
                op=ast.Add(),
                right=low_node,
            )

        node.iter.args = [new_start, up, ast.Constant(value=SIMD_WIDTH)]
        return node

def transform(tree):
    return RewriteSIMDLoops().visit(tree)
