import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class RenameTorchToTriton(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute):
        if node.value.id == 'torch':      
            if node.attr in ['tanh']:
                node.value = new_attr_node(new_name_node('tl'), 'math')
            else:
                node.value = new_name_node('tl')

        elif node.value.id == 'appy' and node.attr not in ['vidx', 'vindex']:
            node.value = new_name_node('tl')
            
        return node

    def visit_Call(self, node):
        if unparse(node).startswith('torch.mv('):
            assert len(node.args) == 2
            newnode = new_attr_call_node(
                    'torch.sum', 
                    [new_mul_node(node.args[0], node.args[1])]
                )
            newnode.lineno = node.lineno
            node = newnode
        
        self.generic_visit(node)
        return node