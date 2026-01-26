import ast_comments as ast
from copy import deepcopy

class RewriteAugAssign(ast.NodeTransformer):
    '''
    This class rewrites AugAssign nodes to Assign nodes, e.g.
    x += 1 is rewritten to x = x + 1
    '''
    def visit_AugAssign(self, node):
        copy_of_target = deepcopy(node.target)
        copy_of_target.ctx = ast.Load()
        
        newnode = ast.Assign(
                targets=[node.target], 
                value=ast.BinOp(
                    left=copy_of_target, 
                    op=node.op, 
                    right=node.value
                ),
                lineno=node.lineno,
            )

        # if isinstance(node.op, ast.Add):
        #     newnode.reduce = '+'
        #     #print('mark as + reduction')
        #     #dump(newnode)
        return newnode