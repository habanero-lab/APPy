import ast

class RewriteFuncCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        from .. import metal_math
        funcname = None
        if isinstance(node.func, ast.Attribute):
            funcname = node.func.attr
        elif isinstance(node.func, ast.Name):
            funcname = node.func.id

        if funcname in ['range', 'prange']:
            return node

        assert hasattr(metal_math, funcname), "Unknown function: " + funcname
        node.func = ast.Name(id=funcname, ctx=ast.Load())
        self.generic_visit(node)
        return node
    
    def visit_For(self, node):
        self.generic_visit(node)
        return node
    
def transform(node):
    return RewriteFuncCalls().visit(node)