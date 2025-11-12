import ast

class RewritePrange(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        
        if isinstance(node.func, ast.Name) and node.func.id == 'prange':
            node.func.id = 'range'
            # Add a keyword argument to indicate parallel=True
            node.keywords.append(ast.keyword(arg='parallel', value=True))
            return node
        elif isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'appy.prange':
            node.func = ast.Name(id='range', ctx=ast.Load())
            # Add a keyword argument to indicate parallel=True
            node.keywords.append(ast.keyword(arg='parallel', value=ast.Constant(value=True)))
            return node
        return node

def transform(tree):
    '''
    This pass rewrites an appy.prange or simply prange call to range(..., parallel=True).
    '''
    return RewritePrange().visit(tree)