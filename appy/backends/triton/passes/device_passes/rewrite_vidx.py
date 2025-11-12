import ast

class RewriteVidx(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'vidx':
            assert len(node.args) == 3
            start, stepsize, bound = node.args
            new_value = ast.BinOp(
                op=ast.Add(),
                left=start,
                right=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='tl', ctx=ast.Load()),
                        attr='arange'
                    ),
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
    `__idx_i = appy.vidx(i, block_size, bound)` 
    to
    `__idx_i = i + tl.arange(0, block_size)`.   
    '''
    return RewriteVidx().visit(tree)