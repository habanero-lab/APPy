import ast

class RewriteTupleAssign(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Tuple):
            assert isinstance(node.targets[0], ast.Tuple)
            nodes = [
                ast.Assign(
                    targets=[target],
                    value=elt
                )
                for target, elt in zip(node.targets[0].elts, node.value.elts)
            ]
            
            return nodes
        else:
            return node


def transform(node):
    '''
    This pass rewrites tuple assignments to multiple assignments, e.g. 
    x, y = 1, 2 is rewritten to x = 1; y = 2
    '''
    return RewriteTupleAssign().visit(node)