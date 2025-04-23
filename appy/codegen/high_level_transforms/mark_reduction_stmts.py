'''
This pass marks reduction statements as such by adding the `reduce` attribute.
A statement is determined to be a reduction if 

1. It has form `x = x + y` or `x = x * y` or `x = max(x, y)` or `x = min(x, y)`
2. The statement is the only definition of `x` inside a loop
3. The defining loop contains only a straight line of statements (i.e., no control flow)
'''

import ast
from appy.ast_utils import *

class GetReductionCandidateStmts(ast.NodeVisitor):
    '''
    If an assignment has form `x = x + y` or `x = x * y` or `x = max(x, y)` or `x = min(x, y)`
    or `x = y + x` or `x = y * x` or `x = max(y, x)` or `x = min(y, x)`, it is a reduction candidate.
    '''
    def __init__(self):
        self.candidates = []

    def visit_Assign(self, node):
        target = node.targets[0]
        value = node.value
        if isinstance(target, ast.Name):
            if isinstance(value, ast.BinOp):
                if isinstance(value.op, ast.Add) or isinstance(value.op, ast.Mult):
                    if isinstance(value.left, ast.Name) and target.id == value.left.id:
                        self.candidates.append(node)
                    elif isinstance(value.right, ast.Name) and target.id == value.right.id:
                        self.candidates.append(node)
            elif isinstance(value, ast.Call):
                if value.func.id == 'max' or value.func.id == 'min':
                    if isinstance(value.args[0], ast.Name) and target.id == value.args[0].id:
                        self.candidates.append(node)
                    elif isinstance(value.args[1], ast.Name) and target.id == value.args[1].id:
                        self.candidates.append(node)
        return node

class MarkReductionStmts(ast.NodeTransformer):
    def visit_For(self, node):
        self.generic_visit(node)
        # Check if the loop has no control flow
        no_control_flow = True
        for stmt in node.body:
            # As long as the loop body contains non-assignment statements
            # it is not a straight line
            if not isinstance(stmt, ast.Assign):
                no_control_flow = False
                break
        
        if no_control_flow:
            visitor = GetReductionCandidateStmts()
            visitor.visit(node)
            # Check if a candidate is the single definition of its target inside the loop
            var_to_candidate = {}
            for candidate in visitor.candidates:
                target = candidate.targets[0]
                if target not in var_to_candidate:
                    var_to_candidate[target] = []
                var_to_candidate[target].append(candidate)

            # Attach the `reduce` attribute for candidates that are single definitions
            for target, candidates in var_to_candidate.items():
                if len(candidates) == 1:
                    stmt = candidates[0]
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
                        if 'parallel_for' in pragma_dict or 'simd' in pragma_dict:
                            if 'reduction' not in pragma_dict:
                                node.pragma_dict['reduction'] = f'{stmt.reduce}:{target.id}'
                                print(f'[DEBUG] pragma_dict: {node.pragma_dict}')
        
        return node
    

def transform(tree):
    return MarkReductionStmts().visit(tree)