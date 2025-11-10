import ast

class AddTritonImports(ast.NodeTransformer):
    '''
    Add imports at the top of the module:
        import torch
        import triton
        import triton.language as tl
    '''
    def visit_Module(self, node):
        import_statements = [
            ast.Import(names=[ast.alias(name='torch', asname=None)]),
            ast.Import(names=[ast.alias(name='triton', asname=None)]),
            ast.Import(names=[ast.alias(name='triton.language', asname='tl')])  
        ]
        node.body = import_statements + node.body
        return node
    
def transform(tree):
    return AddTritonImports().visit(tree)