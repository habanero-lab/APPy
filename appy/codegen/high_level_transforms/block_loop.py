from ast import unparse
from appy.ast_utils import *
from .utils import *
from copy import deepcopy

class BlockLoop(ast.NodeTransformer):
    def __init__(self):
        self.replace_with_sum = False

    def visit_For(self, node: ast.For):    
        if hasattr(node, 'pragma'):
            pragma = node.pragma
            #print('pragma:', pragma)
            if ' block' in pragma:
                block_size = get_pragma_property(pragma, 'block')
                assert block_size, "Block size argument required!"
                loop_idx = node.target
                upper = node.iter.args[1]
                vidx_stmt = f'{loop_idx.id} = vidx({loop_idx.id}, {block_size}, {unparse(upper)})'
                node.body.insert(0, to_ast_node(vidx_stmt))
                node.iter.args[2] = to_ast_expr(block_size)
                self.replace_with_sum = True
        
        self.generic_visit(node)
        self.replace_with_sum = False
        return node

    def visit_Assign(self, node):
        if hasattr(node, 'reduce') and self.replace_with_sum:
            assert node.reduce == '+'
            assert isinstance(node.value, ast.BinOp)
            lhs = unparse(node.targets[0])
            node.value.right = to_ast_expr(f'appy.sum({unparse(node.value.right)})')            
            return node
        else:
            return node