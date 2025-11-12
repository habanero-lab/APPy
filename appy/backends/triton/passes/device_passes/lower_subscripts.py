import ast

class LowerSubscripts(ast.NodeTransformer):
    def __init__(self):
        self.assign_value = None
        self.tl_store = None

    def visit_Assign(self, node):
        self.visit(node.value)
        # This may be used when generating tl.store
        self.assign_value = node.value
        self.visit(node.targets[0])
        
        # Replace the node with tl.store if it exists
        if self.tl_store is not None:
            node = ast.Expr(value=self.tl_store)
            self.tl_store = None
        return node
    
    def visit_Subscript(self, node):        
        self.generic_visit(node)
        if hasattr(node, 'mask'):
            assert isinstance(ast.parse(node.mask).body[0], ast.Expr)
        mask = ast.parse(node.mask).body[0].value if hasattr(node, 'mask') else ast.Constant(value=None)
        # Compute the load or store address
        addr = ast.BinOp(
            op=ast.Add(),
            left=node.value,
            right=node.slice
        )
        if isinstance(node.ctx, ast.Load):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='load'
                ),
                args=[addr],
                keywords=[ast.keyword(arg='mask', value=mask)]
            )
        elif isinstance(node.ctx, ast.Store):
            assert self.assign_value is not None
            val = self.assign_value
            self.assign_value = None # reset
            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='store'
                ),
                args=[addr, val],
                keywords=[ast.keyword(arg='mask', value=mask)]
            )
            self.tl_store = call
            return call
        
    
def transform(node):
    '''
    This pass transforms array subscripts to calls to tl.load or tl.store.
    '''
    return LowerSubscripts().visit(node)