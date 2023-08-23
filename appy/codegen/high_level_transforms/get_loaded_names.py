import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy
import appy.codegen.typesys as typesys

class GetLoadedNames(ast.NodeVisitor):
    def __init__(self, names):
        self.names = names
        self.stored_vars = []

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name):            
            ndim = 1
            if isinstance(node.slice, ast.Tuple):
                ndim = len(node.slice.elts)
            self.names[node.value.id] = ('tensor', ndim)
            #print(self.names)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if unparse(node).startswith('vidx(') or unparse(node).startswith('appy.vidx('):
            if isinstance(node.args[1], ast.Name):
                self.names[node.args[1].id] = ('const', 0)
        self.generic_visit(node)

    def visit_Name(self, node):        
        if node.id.startswith('_top_var'):
            return
        
        if node.id in ['range', 'vidx', 'float', 'int', 'tl', 'torch', 'appy']:
            return

        if isinstance(node.ctx, ast.Store):
            self.stored_vars.append(node.id)         
        
        if node.id not in self.stored_vars and (node.id not in self.names):
            self.names[node.id] = ('scalar', 0)