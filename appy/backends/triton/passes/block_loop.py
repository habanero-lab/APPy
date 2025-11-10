import ast

class BlockLoop(ast.NodeTransformer):
    def __init__(self, pragma):
        self.pragma = pragma
        self.verbose = True

    def get_block_size(self):
        return 256  # Use a fixed block size for now

    def visit_For(self, node):
        if self.verbose:
            print(f'[Block Loop] Got pragma: {self.pragma}')
        
        if 'simd' in self.pragma:
            iter_start, iter_end, iter_step = node.iter.args
            block_size = self.get_block_size()
            assign = ast.Assign(
                targets=[ast.Name(id=node.target.id, ctx=ast.Store())], 
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="appy", ctx=ast.Load()),
                        attr="vidx",
                        ctx=ast.Load()
                    ),
                    args=[
                        iter_start,
                        ast.Constant(block_size),
                        iter_end
                    ],
                    keywords=[]
                )
            )
            node.body.insert(0, assign)
            # Set the loop step to the block size
            node.iter.args[2] = ast.Constant(block_size)

        self.generic_visit(node)
        return node

def transform(tree, pragma):
    '''
    This pass rewrites a for-loop to blocked form, which consists of two steps:
    1. assign the loop index to a call to vidx(...) and insert the assignment in the beginning of the loop body.
    2. rewrite the loop step to be the block size

    The vidx call takes 3 arguments: the loop inde (start), the block size, and the upper bound of the loop.
    '''
    return BlockLoop(pragma).visit(tree)