import ast

def is_float_inf_call(node):
    # must be a call
    if not isinstance(node, ast.Call):
        return False

    # function must be float(...)
    if not (isinstance(node.func, ast.Name) and node.func.id == "float"):
        return False

    # must have exactly 1 positional argument
    if not (len(node.args) == 1 and isinstance(node.args[0], ast.Constant)):
        return False

    arg = node.args[0].value

    # Accept both str and repr-style matches
    return arg in ("inf", "-inf", "+inf", "Infinity", "+Infinity", "-Infinity")


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
        
    def visit_Call(self, node):
        if is_float_inf_call(node):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='full',
                    ctx=ast.Load()
                ),
                args=[
                    ast.Tuple(elts=[], ctx=ast.Load()), 
                    node        # original float("inf") node
                ],
                keywords=[
                    ast.keyword(
                        arg='dtype',
                        value=ast.Attribute(
                            value=ast.Name(id='tl', ctx=ast.Load()),
                            attr='float64',
                            ctx=ast.Load()
                        )
                    )
                ]
            )
        else:
            return self.generic_visit(node)

        
def transform(tree):
    return LowerConstants().visit(tree)