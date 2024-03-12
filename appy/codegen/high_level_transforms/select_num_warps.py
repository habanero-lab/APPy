from ast import unparse
from appy.ast_utils import *
from .utils import *

class InspectAssign(ast.NodeVisitor):
    def __init__(self) -> None:
        self.vec_length = 1
        self.has_reduction = False

    def visit_Assign(self, node):
        if hasattr(node, 'pragma'):
            pragma = node.pragma
            slice_map = parse_pragma(pragma)
            for k, props in slice_map.items():
                if props['single_block']:
                    if props['block'] > self.vec_length:
                        self.vec_length = props['block']
                if props['reduce']:
                    self.has_reduction = props['reduce']
           

class SelectNumWarps(ast.NodeTransformer):
    def visit_For(self, node: ast.For):    
        if hasattr(node, 'pragma'):
            visitor = InspectAssign()
            visitor.visit(node)
            
            if visitor.vec_length >= 1024 and not visitor.has_reduction:
                node.pragma += ' num_warps(8)'
        return node