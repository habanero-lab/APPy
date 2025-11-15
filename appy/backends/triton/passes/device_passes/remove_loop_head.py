import ast

class InsertProgramID(ast.NodeTransformer):
    def get_loop_property(self, loop: ast.For):
        '''
        Extract the keyword arguments of the range call as a dict and return it.
        '''
        for kw in loop.iter.keywords:
            assert isinstance(kw.value, ast.Constant)

        prop = {kw.arg: kw.value.value for kw in loop.iter.keywords}
        return prop
        
    def get_loop_pragma(self, node):
        if hasattr(node, 'pragma'):
            return node.pragma
        else:
            return {}
    
    def visit_For(self, node):
        #prop = self.get_loop_property(node)
        prop = self.get_loop_pragma(node)
        _, _, step = node.iter.args
        if 'parallel_for' in prop:
            assign = ast.Assign(
                targets=[ast.Name(id=node.target.id, ctx=ast.Store())], 
                value=ast.BinOp(
                    op=ast.Mult(),
                    left=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="tl", ctx=ast.Load()),
                            attr="program_id",
                            ctx=ast.Load()
                        ),
                        args=[ast.Constant(0)],
                        keywords=[]
                    ),
                    right=step
                )
            )
            node.body.insert(0, assign)
            return node.body
        else:
            return node


def transform(tree):
    '''
    This pass inserts an assignment `tl.program_id(0) * step` to the loop index at the beginning of the loop body.
    '''
    return InsertProgramID().visit(tree)