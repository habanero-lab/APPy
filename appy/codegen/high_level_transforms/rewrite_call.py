import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class RewriteAPPyCall(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute):
        # Format of node is {value}.{attr}
        if not isinstance(node.value, ast.Name):
            return node

        if node.value.id in ['appy', 'torch', 'cupy']:
            if node.attr in ['vidx', 'vindex']:
                return node
            elif node.attr in ['tanh', 'asin']:
                node.value = new_attr_node(new_name_node('tl'), 'math')
            else:
                node.value = new_name_node('tl')


        # if node.value.id == 'torch':      
        #     if node.attr in ['tanh']:
        #         node.value = new_attr_node(new_name_node('tl'), 'math')
        #     else:
        #         node.value = new_name_node('tl')

        # elif node.value.id == 'appy' and node.attr not in ['vidx', 'vindex']:
        #     node.value = new_name_node('tl')
            
        return node

    def visit_Call(self, node: ast.Call):
        if unparse(node).startswith('appy.mv('):
            assert len(node.args) == 2            
            newnode = new_attr_call_node(
                    'appy.sum', 
                    [new_mul_node(node.args[0], node.args[1])],
                    #keywords={'axis': to_ast_expr('1')}
                )
            newnode.lineno = node.lineno
            node = newnode
        elif unparse(node).startswith('appy.dot('):
            # Rewrite 1D vector dot product
            assert len(node.args) == 2
            newnode = new_attr_call_node(
                    'appy.sum', 
                    [new_mul_node(node.args[0], node.args[1])],
                )
            newnode.lineno = node.lineno
            node = newnode

        elif unparse(node).startswith('appy.atomic_'):
            # Add the first and the second argument
            assert len(node.args) == 3
            new_args = [
                new_add_node(node.args[0], node.args[1]),
                node.args[2],
            ]
            node.args = new_args
        elif unparse(node.func) in ['appy.flip']:
            # Convert "appy.flip(A[..])" to "A[..]", and add a new attr "flip" to its slice
            assert len(node.args) == 1
            #dump(node)
            assert isinstance(node.args[0], ast.Subscript) and isinstance(node.args[0].slice, ast.Slice)
            node = node.args[0]
            node.slice.flip = True
            
        
        self.generic_visit(node)
        return node