import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class RewriteRange(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'range':
            if len(node.args) == 1:
                up = node.args[0]
                node.args = [
                    new_const_node(0),
                    up,
                    new_const_node(1)
                ]
            elif len(node.args) == 2:
                low, up = node.args
                node.args = [
                    low,
                    up,
                    new_const_node(1)
                ]

            assert len(node.args) == 3, ast.dump(node)
        return node