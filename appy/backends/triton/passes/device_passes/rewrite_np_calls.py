import ast

class RewriteNumpyCalls(ast.NodeTransformer):
    def __init__(self):
        self.triton_core_funcs = [
            "abs",
            "cdiv",
            "ceil",
            "clamp",
            "cos",
            "div_rn",
            "erf",
            "exp",
            "exp2",
            "fdiv",
            "floor",
            "fma",
            "log",
            "log2",
            "maximum",
            "minimum",
            "rsqrt",
            "sigmoid",
            "sin",
            "softmax",
            "sqrt",
            "sqrt_rn",
            "umulhi",

            "where",
        ]


    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.triton_core_funcs:
            node.func.value = ast.Name(id='tl', ctx=ast.Load())
        self.generic_visit(node)
        return node
        

def transform(tree):
    '''
    This pass rewrites numpy library calls into their corresponding Triton library calls.
    For example,
    `np.minimum(a, b)` is rewritten to `tl.minimum(a, b)`
    `np.sin(a)` is rewritten to `tl.sin(a)`
    '''
    return RewriteNumpyCalls().visit(tree)