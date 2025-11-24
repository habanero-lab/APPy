import ast
from . import tl_libdevice as libdevice

class RewriteNumpyCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and hasattr(libdevice, node.func.attr):
            node.func.value = ast.Name(id='libdevice', ctx=ast.Load())
        elif isinstance(node.func, ast.Name) and hasattr(libdevice, node.func.id):
            node.func = ast.Attribute(
                value=ast.Name(id='libdevice', ctx=ast.Load()),
                attr=node.func.id
            )
        elif isinstance(node.func, ast.Name) and node.func.id in ["range", "min", "max", "abs", "sum", "pow"]:
            pass # Built-in Python functions
        else:
            if not (isinstance(node.func, ast.Attribute) and node.func.value.id == "tl"):
                raise ValueError(f"Unknown function: {ast.unparse(node.func)}")
        self.generic_visit(node)
        return node
        

def transform(tree):
    '''
    This pass rewrites numpy/math library calls into their corresponding Triton libdevice calls.
    For example,
    `np.log(a)` is rewritten to `libdevice.log(a)`
    `np.sin(a)` is rewritten to `libdevice.sin(a)`
    '''
    return RewriteNumpyCalls().visit(tree)