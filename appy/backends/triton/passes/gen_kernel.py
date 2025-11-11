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

    def gen_triton_decorator(self):
        self.func.decorator_list.append(ast.Attribute(
            value=ast.Name(id='triton', ctx=ast.Load()),
            attr='jit',
            ctx=ast.Load(),
        ))

    def visit_For(self, node):
        from .kernel_passes import rewrite_vidx
        from .kernel_passes import attach_mask_info
        from .kernel_passes import lower_subscripts
        
        attach_mask_info.visit(node)
        node = rewrite_vidx.transform(node)
        node = lower_subscripts.transform(node)
        # Let the transformed loop body be the kernel function body
        self.func.body = node.body
        return node

    def visit_Module(self, node):
        self.init_func()
        self.gen_func_params()        
        self.gen_triton_decorator()
        self.generic_visit(node)
        node.body = [self.func] + node.body
        return node
    
def transform(node, val_map, metadata):
    return GenKernel(val_map, metadata).visit(node)