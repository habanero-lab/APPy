import ast_comments as ast

class RewriteTensorOperation(ast.NodeTransformer):
    def visit_Assign(self, node):
        if hasattr(node, 'pragma'):
            pragma = node.pragma
            dump(node)
            exit(1)