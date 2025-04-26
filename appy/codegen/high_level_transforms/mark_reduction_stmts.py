'''
This pass marks reduction statements as such by adding the `reduce` attribute.
A statement is determined to be a reduction if 

1. It has form `x = x + y` or `x = x * y` or `x = max(x, y)` or `x = min(x, y)`
2. The statement is the only definition of `x` inside a loop
3. The defining loop contains only a straight line of statements (i.e., no control flow)
'''

import ast
from appy.ast_utils import *


class GetDefinitions(ast.NodeVisitor):
    def __init__(self):
        self.definitions = {}

    def visit_Assign(self, node):
        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id not in self.definitions:
                self.definitions[target.id] = []
            self.definitions[target.id].append(node)
        return node
    

class HasBreakOrContinue(ast.NodeVisitor):
    def __init__(self):
        self.has_break = False
        self.has_continue = False

    def visit_Break(self, node):
        self.has_break = True
        return node

    def visit_Continue(self, node):
        self.has_continue = True
        return node


class MarkReductionStmts(ast.NodeTransformer):
    def is_candidate(self, node):
        assert isinstance(node, ast.Assign)
        target = node.targets[0]
        value = node.value
        if isinstance(target, ast.Name):
            if isinstance(value, ast.BinOp):
                if isinstance(value.op, ast.Add) or isinstance(value.op, ast.Mult):
                    if isinstance(value.left, ast.Name) and target.id == value.left.id:
                        return True
                    elif isinstance(value.right, ast.Name) and target.id == value.right.id:
                        return True
            elif isinstance(value, ast.Call):
                if value.func.id == 'max' or value.func.id == 'min':
                    if isinstance(value.args[0], ast.Name) and target.id == value.args[0].id:
                        return True
                    elif isinstance(value.args[1], ast.Name) and target.id == value.args[1].id:
                        return True

        return False

    def visit_For(self, node):
        self.generic_visit(node)
        visitor = HasBreakOrContinue()
        visitor.visit(node)
        if visitor.has_break or visitor.has_continue:
            return node
        
        # Perform reduction analysis if the loop has no break or continue
        def_visitor = GetDefinitions()
        def_visitor.visit(node)
        defs = def_visitor.definitions
        for stmt in node.body:
            # If the assignment statement is the only definition of its target
            # and it is a reduction candidate, mark it as a reduction
            if isinstance(stmt, ast.Assign) and self.is_candidate(stmt) \
                and len(defs[stmt.targets[0].id]) == 1:
                # Attach the `reduce` attribute depending on the operator
                if isinstance(stmt.value, ast.BinOp):
                    if isinstance(stmt.value.op, ast.Add):
                        stmt.reduce = '+'
                    elif isinstance(stmt.value.op, ast.Mult):
                        stmt.reduce = '*'
                elif isinstance(stmt.value, ast.Call):
                    if stmt.value.func.id == 'max':
                        stmt.reduce = 'max'
                    elif stmt.value.func.id == 'min':
                        stmt.reduce = 'min'
                else:
                    assert False

                # If the loop is a parallel for loop or a simd loop, attach the `reduction` clause
                if hasattr(node, 'pragma'):
                    pragma_dict = node.pragma_dict
                    target = stmt.targets[0]
                    if 'parallel_for' in pragma_dict or 'simd' in pragma_dict:
                        if 'reduction' not in pragma_dict:
                            node.pragma_dict['reduction'] = f'{stmt.reduce}:{target.id}'
                            #print(f'[DEBUG] pragma_dict after mark reduction: {node.pragma_dict}')
    
        return node
    

def transform(tree):
    return MarkReductionStmts().visit(tree)