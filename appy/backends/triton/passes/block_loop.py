import ast

def is_name_or_constant_indexing(node):
    return isinstance(node, ast.Name) or isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)

def to_str(node):
    return ast.unparse(node)


class RewriteReductionAssign(ast.NodeTransformer):
    def __init__(self, reductions):        
        self.reductions = reductions

    def rewrite_reduction_value(self, reduction_op, val):
        if reduction_op == '+':
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='sum',
                    ctx=ast.Load()
                ),
                args=[val],
                keywords=[]
            )
        elif reduction_op == '*':
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='prod',
                    ctx=ast.Load()
                ),
                args=[val],
                keywords=[]
            )
        elif reduction_op == 'max':
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='max',
                    ctx=ast.Load()
                ),
                args=[val],
                keywords=[]
            )
        elif reduction_op == 'min':
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='tl', ctx=ast.Load()),
                    attr='min',
                    ctx=ast.Load()
                ),
                args=[val],
                keywords=[]
            )
        else:
            raise ValueError(f"Unsupported reduction operator: {reduction_op}")
        
    def visit_Assign(self, node):
        target = node.targets[0]
        # Should rewrite if target is a reduction
        if to_str(target) in self.reductions:
            reduction_op = self.reductions[to_str(target)]
            if isinstance(node.value, ast.BinOp):
                assert to_str(node.value.left) == to_str(target)      
                node.value.right = self.rewrite_reduction_value(reduction_op, node.value.right)
            elif isinstance(node.value, ast.Call):
                assert node.value.func.id == reduction_op
                node.value.args[1] = self.rewrite_reduction_value(reduction_op, node.value.args[1])

            if self.verbose:
                print(f"[Block Loop] Rewrote simd reduction to {to_str(node)}")

        return node
    

class BlockLoop(ast.NodeTransformer):
    def __init__(self):        
        self.verbose = False

    def get_block_size(self, pragma):
        return 256  # Use a fixed block size for now
    
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
        
    def rewrite_reduction_value(self, reductions, node):        
        reduction_var_map = {}
        for reduction in reductions:
            op, var = reduction.split(':')
            assert var not in reduction_var_map, f"Duplicate reduction variable: {var}"
            reduction_var_map[var] = op

        visitor = RewriteReductionAssign(reduction_var_map)
        node = visitor.visit(node)
        return node

    def visit_For(self, node):
        #prop = self.get_loop_property(node)
        prop = self.get_loop_pragma(node)
        if self.verbose:
            print(f'[Block Loop] Got pragma: {prop}')        
        
        if 'simd' in prop:
            iter_start, iter_end, iter_step = node.iter.args
            block_size = self.get_block_size(prop)
            new_idx = f'__idx_{node.target.id}'
            assign = ast.Assign(
                targets=[ast.Name(id=node.target.id, ctx=ast.Store())], 
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="appy", ctx=ast.Load()),
                        attr="vidx",
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.Name(id=new_idx, ctx=ast.Load()),
                        ast.Constant(block_size),
                        iter_end
                    ],
                    keywords=[]
                )
            )
            node.body.insert(0, assign)
            # Set the loop step to the block size
            node.iter.args[2] = ast.Constant(block_size)

            # Set the loop index to the new index
            node.target.id = new_idx

            # Rewrite reduction value if reduction is present
            if 'reduction' in prop:
                self.rewrite_reduction_value(prop['reduction'].split(','), node)

        self.generic_visit(node)
        return node

def transform(tree):
    '''
    This pass rewrites a for-loop to blocked form, which consists of two steps:
    1. assign the loop index to a call to vidx(...) and insert the assignment in the beginning of the loop body.
    2. rewrite the loop step to be the block size

    The vidx call takes 3 arguments: the loop index (start), the block size, and the upper bound of the loop.
    '''
    return BlockLoop().visit(tree)