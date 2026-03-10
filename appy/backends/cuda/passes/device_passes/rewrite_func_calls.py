import ast


class RewriteFuncCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id == 'range':
                return node
            elif node.func.id in ['int', 'float']:
                node.func = ast.Name(id=f'({node.func.id})', ctx=ast.Load())
                return node

        from .. import device_func_types
        funcname = None
        if isinstance(node.func, ast.Attribute):
            funcname = node.func.attr
        elif isinstance(node.func, ast.Name):
            funcname = node.func.id

        # Map numpy ufunc names to CUDA equivalents
        numpy_to_cuda = {'minimum': 'min', 'maximum': 'max'}
        if funcname in numpy_to_cuda:
            funcname = numpy_to_cuda[funcname]

        assert hasattr(device_func_types, funcname), "Unknown device function: " + funcname
        node.func = ast.Name(id=funcname, ctx=ast.Load())
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        node.body = [self.visit(child) for child in node.body]
        return node


def transform(node):
    return RewriteFuncCalls().visit(node)
