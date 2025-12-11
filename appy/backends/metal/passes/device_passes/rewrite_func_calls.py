import ast

class RewriteFuncCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        # Handle special built-in functions
        if isinstance(node.func, ast.Name):
            if node.func.id in 'range':
                return node
            elif node.func.id in ['int', 'float']:
                node.func = ast.Name(id=f'static_cast<{node.func.id}>', ctx=ast.Load())
                return node

        from .. import device_func_types
        funcname = None
        if isinstance(node.func, ast.Attribute):
            funcname = node.func.attr
        elif isinstance(node.func, ast.Name):
            funcname = node.func.id

        # if funcname in ['range', 'prange']:
        #     return node

        assert hasattr(device_func_types, funcname), "Unknown function: " + funcname
        node.func = ast.Name(id=funcname, ctx=ast.Load())
        self.generic_visit(node)
        return node
    
    def visit_For(self, node):
        new_body = []
        for child in node.body:
            new_body.append(self.visit(child))
        node.body = new_body
        return node
    
def transform(node):
    return RewriteFuncCalls().visit(node)