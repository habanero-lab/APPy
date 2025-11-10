import ast

class GenKernel(ast.NodeTransformer):
    def visit_Module(self, node):
        kernel_code = ast.parse("""
@triton.jit
def kernel(a, a_shape_0, b, c):
    pass
    i = 0 + tl.program_id(0) * 256
    tl.store(
        c + (i + tl.arange(0, 256) + tl.arange(0, 1)),
        tl.load(
            a + (i + tl.arange(0, 256) + tl.arange(0, 1)),
            mask=i + tl.arange(0, 256) < a_shape_0,
        )
        + tl.load(
            b + (i + tl.arange(0, 256) + tl.arange(0, 1)),
            mask=i + tl.arange(0, 256) < a_shape_0,
        ),
        mask=i + tl.arange(0, 256) < a_shape_0,
    )
        """)
        node.body = kernel_code.body + node.body
        return node
    
def transform(node):
    return GenKernel().visit(node)