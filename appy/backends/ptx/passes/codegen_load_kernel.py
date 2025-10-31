import ast
from pathlib import Path

class AddCodeLoadKernel(ast.NodeTransformer):
    def __init__(self, path):
        self.path = path
        
    def visit_Module(self, node):
        load_kernel_code = ast.parse(f"""
# Load PTX code from file
with open('{self.path}', 'r') as f:
    ptx_code = f.read()

# Load module and get kernel function
module = cuda.module_from_buffer(ptx_code.encode())
kernel = module.get_function("kernel")
        """)
        node.body = load_kernel_code.body + node.body
        return node