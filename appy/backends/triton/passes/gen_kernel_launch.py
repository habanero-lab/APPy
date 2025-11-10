import ast

class GenKernelLaunch(ast.NodeTransformer):
    def __init__(self, val_map):
        self.val_map = val_map

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
        print(self.val_map)
        #arg_nodes = [ast.Name(id=x, ctx=ast.Load()) for x in self.val_map.keys()]
        arg_nodes = []
        for var, val in self.val_map.items():
            ty = type(val)
            if f'{ty.__module__}.{ty.__name__}' == 'numpy.ndarray':
                arg_nodes.append(ast.Name(id=f'__tg_{var}', ctx=ast.Load()))
            else:
                arg_nodes.append(ast.Name(id=var, ctx=ast.Load()))

        kwarg_nodes = [ast.keyword(arg="num_warps", value=ast.Constant(4))]
        assigns.append(ast.Expr(
            value=ast.Call(
                func=ast.Subscript(
                    value=ast.Name(id="kernel", ctx=ast.Load()),
                    slice=ast.Name(id="grid", ctx=ast.Load())
                ),
                args=arg_nodes,
                keywords=kwarg_nodes
            )
        ))
        return assigns
    
def transform(tree, val_map):
    return GenKernelLaunch(val_map).visit(tree)