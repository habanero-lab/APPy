import ast
from pathlib import Path

class AddCodeLoadKernel(ast.NodeTransformer):
    '''
    Add code to load the kernel from PTX file into the AST.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    def visit_Module(self, node):
        ptx_file_path = Path(__file__).parent.parent / 'sample_kernels' / 'vec_add.ptx'
        load_kernel_code = ast.parse(f"""
# Load PTX code from file
with open('{ptx_file_path}', 'r') as f:
    ptx_code = f.read()

# Load module and get kernel function
module = cuda.module_from_buffer(ptx_code.encode())
kernel = module.get_function("vec_add")
        """)
        node.body = load_kernel_code.body + node.body
        return node