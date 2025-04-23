'''
This pass convert ternary expressions to `appy.where` expressions.
'''
import ast
from appy.ast_utils import *

class TernaryToWhere(ast.NodeTransformer):
    def visit_IfExp(self, node: ast.IfExp):
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='appy', ctx=ast.Load()), attr='where', ctx=ast.Load()),
            args=[node.test, node.body, node.orelse],
            keywords=[]
        )
    

def transform(tree):
    return TernaryToWhere().visit(tree)