import ast
from copy import deepcopy

class RewriteAugAssign(ast.NodeTransformer):
    def visit_AugAssign(self, node):
        left = node.target
        right = node.value

        left_copy = deepcopy(left)
        left_copy.ctx = ast.Load()
        newnode = ast.Assign(
                targets=[left], 
                value=ast.BinOp(
                    left=left_copy,
                    op=node.op, 
                    right=right
                ),
                lineno=node.lineno,
            )
        return newnode
    
def transform(node):
    '''
    This pass rewrites AugAssign nodes to Assign nodes, e.g.
    x += 1 is rewritten to x = x + 1
    '''
    return RewriteAugAssign().visit(node)