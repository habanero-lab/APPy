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
            # Built-in Python functions - functions that can be directly used inside Triton kernels
            pass 
        elif isinstance(node.func, ast.Name) and node.func.id == "float":
            # Triton kernel only supports float argument "inf", "-inf" or "nan"
            assert len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and \
                node.args[0].value in ["inf", "-inf", "nan"], ast.unparse(node)
        else:
            # Triton tl.* functions are fine
            if not (isinstance(node.func, ast.Attribute) and node.func.value.id == "tl"):
                print(ast.unparse(node))
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