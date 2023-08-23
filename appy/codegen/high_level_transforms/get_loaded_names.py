import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class GetLoadedNames(ast.NodeVisitor):
    def __init__(self, names):
        self.names = names
        self.stored_vars = []

    def visit_Name(self, node):        
        if node.id.startswith('_top_var'):
            return

        if node.id in ['range', 'vidx', 'APPY_BLOCK', 'float', 'int', 'tl', 'torch']:
            return

        if isinstance(node.ctx, ast.Store):
            self.stored_vars.append(node.id) 
        
        if node.id not in self.stored_vars and (node.id not in self.names):
            self.names.append(node.id)