import ast

class RewriteVidx(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'vidx':
            assert len(node.args) == 3
            start, stepsize, bound = node.args
            new_value = ast.BinOp(
                op=ast.Add(),
                left=ast.BinOp(
                    op=ast.Mult(),
                    left=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                           attr='program_id', ctx=ast.Load()),
                        args=[ast.Constant(0)],
                        keywords=[]
                    ),
                    right=stepsize
                ),
                right=ast.Call(
                    func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                       attr='arange', ctx=ast.Load()),
                    args=[ast.Constant(0), stepsize],
                    keywords=[]
                )
            )
            return new_value
        else:
            self.generic_visit(node)
            return node 
    
def transform(tree):
    '''
    This pass simplifies kernel codegen by rewriting vectorized index assignment like 
    `i = appy.vidx(i, block_size, bound)` 
    to
    `i = tl.program_id(0) * block_size + tl.arange(0, block_size)`.   
    '''
    return RewriteVidx().visit(tree)