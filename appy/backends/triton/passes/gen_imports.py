import ast

class AddTritonImports(ast.NodeTransformer):
    '''
    Add imports at the top of the module:
        import torch
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    '''
    def visit_Module(self, node):
        import_statements = [
            ast.Import(names=[ast.alias(name='torch', asname=None)]),
            ast.Import(names=[ast.alias(name='triton', asname=None)]),
            ast.Import(names=[ast.alias(name='triton.language', asname='tl')]),
            ast.ImportFrom(
                module='triton.language.extra',
                names=[ast.alias(name='libdevice', asname=None)]
            )
        ]
        node.body = import_statements + node.body
        return node
    
def transform(tree):
    return AddTritonImports().visit(tree)