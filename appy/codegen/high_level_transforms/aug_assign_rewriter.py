import ast_comments as ast
from ast_transforms.utils import *

class RewriteAugAssign(ast.NodeTransformer):
    '''
    This class rewrites AugAssign nodes to Assign nodes, e.g.
    x += 1 is rewritten to x = x + 1
    '''
    def visit_AugAssign(self, node):
        left = node.target
        right = node.value
        newnode = ast.Assign(
                targets=[left], 
                value=ast.BinOp(
                    left=deepcopy_ast_node(left, ctx=ast.Load()), 
                    op=node.op, 
                    right=right
                ),
                lineno=node.lineno,
            )

        if isinstance(node.op, ast.Add):
            newnode.reduce = '+'
            #print('mark as + reduction')
            #dump(newnode)
        return newnode