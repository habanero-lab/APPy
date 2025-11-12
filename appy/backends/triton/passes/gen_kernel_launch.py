import ast

class GenKernelLaunch(ast.NodeTransformer):
    def __init__(self, h2d_map, metadata):
        self.val_map = metadata['val_map']
        self.loop_name = metadata['loop_name']
        self.h2d_map = h2d_map
        self.replaced_loop = None    

    def visit_For(self, node):
        iter_start, iter_end, iter_step = node.iter.args
        # Generate grid computation, e.g.
        # grid = ((iter_end - iter_start + iter_step - 1) // iter_step,)
        assigns = []
        assigns.append(ast.Assign(
            targets=[ast.Name(id="grid", ctx=ast.Store())],
            value=ast.Tuple(elts=[ast.BinOp(
                op=ast.FloorDiv(),
                left=ast.BinOp(
                    op=ast.Add(),
                    left=ast.BinOp(
                        op=ast.Sub(),
                        left=iter_end,
                        right=iter_start
                    ),
                    right=ast.BinOp(
                        op=ast.Sub(),
                        left=iter_step,
                        right=ast.Constant(1)
                    )
                ),
                right=iter_step
            )])
        ))

        # Call the triton kernel with arguments        
        arg_nodes = [
            ast.Name(id=self.h2d_map[var], ctx=ast.Load()) if var in self.h2d_map
            else ast.Name(id=var, ctx=ast.Load())
            for var in self.val_map
        ]
        # APPy's convention for kernel name
        kernel_name = '_' + self.loop_name
        assigns.append(ast.Expr(
            value=ast.Call(
                func=ast.Subscript(
                    value=ast.Name(id=kernel_name, ctx=ast.Load()),
                    slice=ast.Name(id="grid", ctx=ast.Load())
                ),
                args=arg_nodes,
                keywords=[ast.keyword(arg="num_warps", value=ast.Constant(4))]
            )
        ))
        self.replaced_loop = node
        return assigns
    
def transform(tree, h2d_map, metadata):
    visitor = GenKernelLaunch(h2d_map, metadata)
    tree = visitor.visit(tree)
    return tree, visitor.replaced_loop