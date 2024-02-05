from ast import unparse
from appy.ast_utils import *
from .utils import *

class ConvertSeqLoop(ast.NodeTransformer):
    def visit_For(self, node: ast.For):    
        if hasattr(node, 'pragma'):
            pragma = node.pragma
            if pragma.strip() == '#pragma sequential for':
                old_node = node
                delattr(old_node, 'pragma')
                node = new_for_loop(
                    target=new_name_node('_', ctx=ast.Store()),
                    low = new_const_node(0),
                    up = new_const_node(1),
                    step = new_const_node(1),
                )
                node.lineno = old_node.lineno
                node.pragma = '#pragma parallel for'
                node.body.append(old_node)
            
        return node