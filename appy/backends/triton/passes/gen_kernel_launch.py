import ast

class GenKernelLaunch(ast.NodeTransformer):
    def __init__(self, val_map, h2d_map, metadata):
        self.val_map = val_map
        self.h2d_map = h2d_map
        self.metadata = metadata

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
        assigns.append(ast.Expr(
            value=ast.Call(
                func=ast.Subscript(
                    value=ast.Name(id=self.metadata['loop_name'], ctx=ast.Load()),
                    slice=ast.Name(id="grid", ctx=ast.Load())
                ),
                args=arg_nodes,
                keywords=[ast.keyword(arg="num_warps", value=ast.Constant(4))]
            )
        ))
        return assigns
    
def transform(tree, val_map, h2d_map, metadata):
    return GenKernelLaunch(val_map, h2d_map, metadata).visit(tree)