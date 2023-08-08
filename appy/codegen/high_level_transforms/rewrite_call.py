import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class RewriteTopCall(ast.NodeTransformer):
    def visit_Attribute(self, node):        
        node.value = new_name_node('tl')
        return node