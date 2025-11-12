import ast

class LowerConstants(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, float):
            # Make float literals float64 by default, e.g. "tl.full((), 0.0, dtype=tl.float64)"
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='full'
                ),
                args=[ast.Tuple(elts=[], ctx=ast.Load()), node],
                keywords=[ast.keyword(arg='dtype', value=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='float64'
                ))]
            )
        else:
            return node
        
def transform(tree):
    return LowerConstants().visit(tree)