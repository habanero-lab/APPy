import ast

def is_for_range_loop(node):
    return isinstance(node, ast.For) and \
        isinstance(node.iter, ast.Call) and \
        isinstance(node.iter.func, ast.Name) and \
        node.iter.func.id == 'range'

class RewriteShaderLoops(ast.NodeTransformer):
    def visit_For(self, node):
        # Make the top-level for-loop a prange for loop
        # Check the first child, if it is a single child and is a for-range, make it a prange as well
        if is_for_range_loop(node):
            node.iter.func.id = 'prange'

            if len(node.body) == 1 and is_for_range_loop(node.body[0]):
                node.body[0].iter.func.id = 'prange'

        return node
    

def transform(node):
    return RewriteShaderLoops().visit(node)