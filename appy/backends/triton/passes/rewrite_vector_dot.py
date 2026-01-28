import ast
from astpass.passes import shape_analysis

class RewriteVectorDot(ast.NodeTransformer):
    def __init__(self, shape_info):
        self.shape_info = shape_info

    def visit_BinOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.MatMult):
            left_shape = self.shape_info[node.left]
            right_shape = self.shape_info[node.right]
            # Rewrite vector dot product into a multiplication followed by a np.sum call
            if len(left_shape) == 1 and len(right_shape) == 1 and left_shape[0] == right_shape[0]:
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='np', ctx=ast.Load()),
                        attr='sum'
                    ),
                    args=[ast.BinOp(
                        op=ast.Mult(),
                        left=node.left,
                        right=node.right
                    )],
                    keywords=[]
                )
            else:
                raise RuntimeError(f"Unsupported matmul expression: {ast.unparse(node)}")
        return node
    
def transform(tree, val_map):
    # A quick fix to add numpy into the scope
    import numpy as np
    shape_info = shape_analysis.analyze(tree, val_map | {'np': np})
    return RewriteVectorDot(shape_info).visit(tree)