import ast

class BlockLoop(ast.NodeTransformer):
    def __init__(self, pragma):
        self.pragma = pragma
        self.verbose = True

    def visit_For(self, node):
        if self.verbose:
            print(f'[Block Loop] Got pragma: {self.pragma}')

def transform(tree, pragma):
    return BlockLoop(pragma).visit(tree)