import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy
from .utils import parse_pragma

class InsertInitialization(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign):
        if hasattr(node, 'pragma') and '=>' in node.pragma:
            slice_map = parse_pragma(node.pragma)
            for key,props in slice_map.items():
                if props['reduce'] and isinstance(node.targets[0], ast.Subscript):
                    op, var = props['reduce'].split(':')
                    node.init_hook = op, var
                    if op == 'sum':
                        return [to_ast_node(f'{var}.fill_(0)'), node]
                    elif op == 'max':
                        return [to_ast_node(f'{var}.fill_(float("-inf"))'), node]
                    elif op == 'min':
                        return [to_ast_node(f'{var}.fill_(float("inf"))'), node]
                    else:
                        assert False
        return node