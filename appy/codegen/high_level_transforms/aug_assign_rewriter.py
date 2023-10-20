import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class RewriteAugAssign(ast.NodeTransformer):
    def visit_AugAssign(self, node):
        left = node.target
        right = node.value
        newnode = ast.Assign(targets=[left], lineno=node.lineno)
        leftcopy = deepcopy(left)
        if isinstance(leftcopy, ast.Subscript):
            leftcopy.ctx = ast.Load()
        newnode.value = ast.BinOp(left=leftcopy, op=node.op, right=node.value)

        if isinstance(node.op, ast.Add):
            newnode.reduce = '+'
            #print('mark as + reduction')
            #dump(newnode)
        return newnode