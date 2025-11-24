import ast

class RewriteMiltiDimensionalSubscripts(ast.NodeTransformer):
    '''
    Rewrites multi-demensional subscripts to single-dimensional subscripts, e.g.
        A[i,j] is rewritten to A[i*A_stride_0 + j]
        A[i,j,k] is rewritten to A[i*A_stride_0 + j*A_stride_1 + k]
    '''
    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.slice, ast.Tuple):
            elts = node.slice.elts
            new_slice = elts[-1]
            for i in range(len(elts)-1):
                new_slice = ast.BinOp(
                    op=ast.Add(),
                    left=ast.BinOp(
                        op=ast.Mult(),
                        left=elts[i],
                        right=ast.Name(id=f'{node.value.id}_stride_{i}', ctx=ast.Load()),
                    ),
                    right=new_slice
                )
            node.slice = new_slice
        return node

class LowerSubscripts(ast.NodeTransformer):
    def __init__(self):
        self.assign_value = None
        self.tl_store = None

    def visit_Assign(self, node):
        node.value = self.visit(node.value)
        # This may be used when generating tl.store
        self.assign_value = node.value
        node.targets[0] = self.visit(node.targets[0])
        
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
                keywords=[ast.keyword(arg='mask', value=mask)] \
                    if hasattr(node, 'mask') else []
                    
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
    node = RewriteMiltiDimensionalSubscripts().visit(node)
    return LowerSubscripts().visit(node)