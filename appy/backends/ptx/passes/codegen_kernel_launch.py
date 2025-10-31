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
    def visit_For(self, node):
        # Assuming the kernel function is named 'kernel' and takes two arguments
        # Replace the pfor loop with kernel launch code
#         launch_code = ast.parse("""
# # Define grid and block dimensions
# block_size = 256
# grid_size = (N + block_size - 1) // block_size
# # Launch the kernel
# kernel(a_gpu, b_gpu, c_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))
#         """)   
        # Launch N thread blocks and 1 thread in each block
        launch_code = ast.parse("""
N = a.shape[0]
block_size = 1
grid_size = N
kernel(a_gpu, b_gpu, c_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))
        """)
        return launch_code.body