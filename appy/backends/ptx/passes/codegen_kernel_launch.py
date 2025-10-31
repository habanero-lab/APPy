import ast

class CodegenKernelLaunch(ast.NodeTransformer):
    '''
    Add code to launch the kernel in the AST.

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
        launch_code = ast.parse("""
N = a.shape[0]
block_size = 1
grid_size = N
kernel(a_gpu, np.int32(N), b_gpu, c_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))
        """)
        node.body = launch_code.body + node.body
        return node