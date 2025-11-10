import ast

class GenKernel(ast.NodeTransformer):
    def __init__(self, val_map, metadata):
        self.val_map = val_map
        self.metadata = metadata
        self.func = None

    def init_func(self):
        self.func = ast.FunctionDef(
            name=self.metadata['loop_name'],
            args=ast.arguments(
                args=[],
                vararg=None,
                kwarg=None,
                defaults=[],
            ),
            body=[],
        )

    def gen_func_params(self):
        for k, v in self.val_map.items():
            self.func.args.args.append(ast.arg(arg=k))

    def gen_func_body(self):
        # Let's use the hardcoded kernel code for now
        kernel_code = ast.parse("""
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
        self.func.body = kernel_code.body

    def gen_triton_decorator(self):
        self.func.decorator_list.append(ast.Attribute(
            value=ast.Name(id='triton', ctx=ast.Load()),
            attr='jit',
            ctx=ast.Load(),
        ))

    def visit_Module(self, node):
        self.init_func()
        self.gen_func_params()
        self.gen_func_body()
        self.gen_triton_decorator()
        node.body = [self.func] + node.body
        return node
    
def transform(node, val_map, metadata):
    return GenKernel(val_map, metadata).visit(node)