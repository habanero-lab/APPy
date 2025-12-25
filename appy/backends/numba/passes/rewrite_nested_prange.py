import ast

class RewriteNestedPrange(ast.NodeTransformer):
    def __init__(self):
        self.loop_index = 0

    def get_new_loop_index(self):
        self.loop_index += 1
        return f"__collapsed_idx_{self.loop_index}"
    
    def is_iter_prange(self, iter: ast.Call):
        return isinstance(iter, ast.Call) and ast.unparse(iter.func) in ["prange", "appy.prange"]
    
    def visit_For(self, node):        
        # If this for is prange and its first child is also prange, we do transform
        if self.is_iter_prange(node.iter):
            firstchild = node.body[0]
            if isinstance(firstchild, ast.For) and self.is_iter_prange(firstchild.iter):
                assert len(node.iter.args) == 1 and len(firstchild.iter.args) == 1, f"Only prange for loops with 1 argument are supported, got: {ast.unparse(node)}"
                outer_id = node.target.id
                inner_id = firstchild.target.id

                outer_bound = node.iter.args[0]
                inner_bound = firstchild.iter.args[0]
                
                new_idx = self.get_new_loop_index()
                new_loop = ast.For(
                    target=ast.Name(id=new_idx, ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="appy", ctx=ast.Load()),
                            attr="prange",
                            ctx=ast.Load()
                        ),
                        args=[ast.BinOp(
                            op=ast.Mult(),
                            left=outer_bound,
                            right=inner_bound
                        )],
                        keywords=[]
                    ),
                    body=[],
                    orelse=[]
                )

                new_loop.body = [
                    ast.Assign(
                        targets=[ast.Name(id=outer_id, ctx=ast.Store())],
                        value=ast.BinOp(
                            op=ast.FloorDiv(),
                            left=ast.Name(id=new_idx, ctx=ast.Load()),
                            right=inner_bound
                        ), 
                        lineno=node.lineno
                    ),
                    ast.Assign(
                        targets=[ast.Name(id=inner_id, ctx=ast.Store())],
                        value=ast.BinOp(
                            op=ast.Mod(),
                            left=ast.Name(id=new_idx, ctx=ast.Load()),
                            right=inner_bound
                        ),
                        lineno=node.lineno
                    ),
                    *firstchild.body
                ]
                
                return new_loop
            
        return node
    
def transform(tree: ast.Module):
    return RewriteNestedPrange().visit(tree)