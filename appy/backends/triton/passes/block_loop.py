import ast

class BlockLoop(ast.NodeTransformer):
    def __init__(self):        
        self.verbose = 1

    def get_block_size(self):
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

    def visit_For(self, node):
        #prop = self.get_loop_property(node)
        prop = self.get_loop_pragma(node)
        if self.verbose:
            print(f'[Block Loop] Got pragma: {prop}')        
        
        if 'simd' in prop:
            iter_start, iter_end, iter_step = node.iter.args
            block_size = self.get_block_size()
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