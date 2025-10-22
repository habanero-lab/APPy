import ast

class ReplacePForWithKernelLaunchStub(ast.NodeTransformer):
    '''
    Transforms both "for i in prange(...)" and "for i in appy.prange(...)" loops into:

        appy._kernel_launch(
            loop_ast=<loop source>,
            loop_name="kernel_loop_1",
            scope=locals(),
            global_scope=globals()
        )
    '''
    def __init__(self):
        super().__init__()
        self.loop_counter = 0

    def visit_For(self, node):
        self.generic_visit(node)

        # --- check for "prange" or "appy.prange"
        is_prange = (
            isinstance(node.iter, ast.Call)
            and (
                (isinstance(node.iter.func, ast.Name) and node.iter.func.id == "prange")
                or (
                    isinstance(node.iter.func, ast.Attribute)
                    and isinstance(node.iter.func.value, ast.Name)
                    and node.iter.func.value.id == "appy"
                    and node.iter.func.attr == "prange"
                )
            )
        )

        if is_prange:
            self.loop_counter += 1
            loop_name = f"kernel_loop_{self.loop_counter}"
            loop_source = ast.unparse(node)

            new_call = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="appy", ctx=ast.Load()),
                        attr="_kernel_launch",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=loop_source),
                        ast.Constant(value=loop_name),
                        ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                        ast.Call(func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[]),
                    ],
                    keywords=[],
                ),
            )


            ast.fix_missing_locations(new_call)

            return new_call

        return node

    
def transform(node):
    return ReplacePForWithKernelLaunchStub().visit(node)