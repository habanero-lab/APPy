'''
This pass convert ternary expressions to `appy.where` expressions.
'''
import ast
from appy.ast_utils import *

class TernaryToWhere(ast.NodeTransformer):
    def __init__(self):
        self.count = 0
        self.extra_stmts = []

    def visit_Assign(self, node):
        self.generic_visit(node)
        if self.extra_stmts:
            new_nodes = self.extra_stmts + [node]
            self.extra_stmts = []
            return new_nodes
        else:
            return node

    def visit_IfExp(self, node: ast.IfExp):
        # If the test is not a Name node, we create a new temporary condition variable
        # assign the test to the variable, and use that variable in the where expression
        if not isinstance(node.test, ast.Name):
            assign = ast.Assign(
                targets=[ast.Name(id=f'__cond_{self.count}', ctx=ast.Store())],
                value=node.test,
                lineno=node.lineno
            )
            where_expr = ast.Call(
                func=ast.Attribute(value=ast.Name(id='appy', ctx=ast.Load()), attr='where', ctx=ast.Load()),
                args=[
                    ast.Name(id=f'__cond_{self.count}', ctx=ast.Load()),
                    node.body,
                    node.orelse
                ],
                keywords=[]
            )
            self.count += 1
            self.extra_stmts.append(assign)
            return where_expr
        # Otherwise we just create a where expression
        else:
            where_expr = ast.Call(
                func=ast.Attribute(value=ast.Name(id='appy', ctx=ast.Load()), attr='where', ctx=ast.Load()),
                args=[
                    node.test,
                    node.body,
                    node.orelse
                ],
                keywords=[]
            )
            return where_expr
        


def transform(tree):
    return TernaryToWhere().visit(tree)