import ast_comments as astc
from ..utils import parse_pragma

class AttachPragma(astc.NodeVisitor):
    def __init__(self):
        self.pragma = None

    def visit_Comment(self, node):
        if node.value.startswith('#pragma '):
            self.pragma = node.value
        return node
    
    def visit_For(self, node):
        if self.pragma:
            node.pragma = parse_pragma(self.pragma)
            self.pragma = None
        self.generic_visit(node)
        return node
    
    def visit_Assign(self, node):
        if self.pragma:
            node.pragma = parse_pragma(self.pragma)
            self.pragma = None
        return node
    

def visit(node):
    return AttachPragma().visit(node)