import ast

class RewriteTernary(ast.NodeTransformer):
    def visit_IfExp(self, node):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='tl', ctx=ast.Load()),
                attr='where',
                ctx=ast.Load()
            ),
            args=[node.test, node.body, node.orelse],
            keywords=[]
        )



def transform(tree):
    '''
    This pass rewrites a ternary operator into a tl.where call.
    '''
    return RewriteTernary().visit(tree)