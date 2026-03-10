import ast
from ..constants import BLOCK_SIZE


class RewriteSIMDLoops(ast.NodeTransformer):
    '''
    Rewrites SIMD-annotated inner loops into strided loops over threadIdx.x.

    A loop like:
        for j in range(low, up):  # pragma simd
    becomes:
        for j in range(lane + low, up, BLOCK_SIZE):

    Each thread handles a strided subset of iterations starting at its lane index.
    '''
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

        lane = ast.Name(id='lane', ctx=ast.Load())
        if low_node is None:
            new_start = lane
        else:
            new_start = ast.BinOp(left=lane, op=ast.Add(), right=low_node)

        node.iter.args = [new_start, up, ast.Constant(value=BLOCK_SIZE)]
        return node


def transform(tree):
    return RewriteSIMDLoops().visit(tree)
