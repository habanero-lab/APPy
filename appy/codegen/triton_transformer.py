import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
import random
import re

class TritonKernelTransformer(ast.NodeTransformer):
    def __init__(self, grid):
        self.grid = grid
        self.block_dim = 0

    def get_params_from_loop(node: ast.For):
        target = node.target
        low, up, step = node.iter.args
        return target, low, up, step

    def visit_For(self, node):
        if hasattr(node, 'pragma'):            
            index_var = node.target
            low, up, step = node.iter.args
            pid_stmt = new_assign_node(
                index_var, 
                new_add_node(
                    low,
                    new_attr_call_node('tl.program_id', [new_const_node(self.block_dim)])
                )
            )
            self.block_dim += 1
            # If loop is parallel, return only its body (no explicit loop)
            self.generic_visit(node)
            return node.body
        else:
            self.generic_visit(node)
            return node

    def visit_Subscript(self, node: ast.Subscript):
        dump(node)
        print('in subscript')
        exit(1)

    def visit_Assign(self, node: ast.Assign):
        ast.NodeTransformer.generic_visit(self, node)
        #self.generic_visit(node)
        print('in assign')
        dump(node)
        exit(1)
