import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class RenameTorchToTriton(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute):
        if not isinstance(node.value, ast.Name):
            return node

        if node.value.id == 'torch':      
            if node.attr in ['tanh']:
                node.value = new_attr_node(new_name_node('tl'), 'math')
            else:
                node.value = new_name_node('tl')

        elif node.value.id == 'appy' and node.attr not in ['vidx', 'vindex']:
            node.value = new_name_node('tl')
            
        return node

    def visit_Call(self, node: ast.Call):
        if unparse(node).startswith('torch.mv('):
            assert len(node.args) == 2
            newnode = new_attr_call_node(
                    'torch.sum', 
                    [new_mul_node(node.args[0], node.args[1])]
                )
            newnode.lineno = node.lineno
            node = newnode

        if unparse(node).startswith('appy.atomic_'):
            # Add the first and the second argument
            assert len(node.args) == 3
            new_args = [
                new_add_node(node.args[0], node.args[1]),
                node.args[2],
            ]
            node.args = new_args
        
        self.generic_visit(node)
        return node