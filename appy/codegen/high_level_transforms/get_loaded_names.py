import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy
import appy.codegen.typesys as typesys

class ExtractArguments(ast.NodeVisitor):
    def __init__(self, names):
        self.names = names
        self.stored_vars = []
        self.func_or_package_names = []

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name):            
            ndim = 1
            if isinstance(node.slice, ast.Tuple):
                ndim = len(node.slice.elts)
            self.names[node.value.id] = ('tensor', ndim)
            #print(self.names)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            self.func_or_package_names.append(node.func.id)

        if unparse(node).startswith('vidx(') or unparse(node).startswith('appy.vidx('):
            if isinstance(node.args[1], ast.Name):
                self.names[node.args[1].id] = ('const', 0)
        
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            self.func_or_package_names.append(node.value.id)

        self.generic_visit(node)

    def visit_Name(self, node):   
        id = node.id     
        if id.startswith('_top_var'):
            return

        if isinstance(node.ctx, ast.Store):
            self.stored_vars.append(id)         
        
        if id not in (self.stored_vars + self.func_or_package_names) and (id not in self.names):
            self.names[id] = ('scalar', 0)