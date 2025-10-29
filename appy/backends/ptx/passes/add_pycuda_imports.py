import ast

class AddPyCUDAImports(ast.NodeTransformer):
    '''
    Add the following import statement at the top of the AST:
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cuda
        import pycuda.autoinit 
    '''
    def visit_Module(self, node):
        import_statements = [
            ast.Import(names=[ast.alias(name='pycuda.gpuarray', asname='gpuarray')]),
            ast.Import(names=[ast.alias(name='pycuda.driver', asname='cuda')]),
            ast.Import(names=[ast.alias(name='pycuda.autoinit', asname=None)])
        ]
        node.body = import_statements + node.body
        return node
    
