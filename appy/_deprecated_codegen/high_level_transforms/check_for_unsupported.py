import ast

class HasBreakOrContinueOrReturn(ast.NodeVisitor):
    def __init__(self):
        self.has_break = False
        self.has_continue = False
        self.has_return = False

    def visit_Break(self, node):
        self.has_break = True
        return node

    def visit_Continue(self, node):
        self.has_continue = True
        return node

    def visit_Return(self, node):
        self.has_return = True
        return node


class CheckForUnsupported(ast.NodeVisitor):
    '''
    Check for each parallel for region, the following constructs are unsupported:
        * break, continue and return
    '''
    def visit_For(self, node):
        if not hasattr(node, 'pragma'):
            self.generic_visit(node)
            return node
        
        if 'parallel_for' in node.pragma_dict:
            visitor = HasBreakOrContinueOrReturn()
            visitor.visit(node)
            if visitor.has_break or visitor.has_continue or visitor.has_return:
                raise Exception(f'break, continue and return are not supported in parallel for')


def transform(node):
    CheckForUnsupported().visit(node)
    return node