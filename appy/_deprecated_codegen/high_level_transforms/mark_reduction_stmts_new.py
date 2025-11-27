'''
This pass marks reduction statements as such by adding the `reduce` attribute.
A statement is determined to be a reduction if 

1. It has form `x = x + y` or `x = x * y` or `x = max(x, y)` or `x = min(x, y)`
2. The statement is the only definition of `x` inside a loop
3. The defining loop contains only a straight line of statements (i.e., no control flow)
'''

import ast
from appy.ast_utils import *


class GetAssignReductionOps(ast.NodeVisitor):
    def __init__(self):
        self.reduction_ops = {}

    def get_reduction_op(self, node):
        '''
        Return the reduction operator for `node` if `node` is a reduction candidate, otherwise return None
        '''
        assert isinstance(node, ast.Assign)
        target = node.targets[0]
        value = node.value
        if isinstance(target, ast.Name):
            if isinstance(value, ast.BinOp):
                if isinstance(value.op, ast.Add) or isinstance(value.op, ast.Mult):
                    op = '+' if isinstance(value.op, ast.Add) else '*'
                    if isinstance(value.left, ast.Name) and target.id == value.left.id:
                        return op
                    elif isinstance(value.right, ast.Name) and target.id == value.right.id:
                        return op
            elif isinstance(value, ast.Call):
                if value.func.id == 'max' or value.func.id == 'min':
                    op = value.func.id
                    if isinstance(value.args[0], ast.Name) and target.id == value.args[0].id:
                        return op
                    elif isinstance(value.args[1], ast.Name) and target.id == value.args[1].id:
                        return op

        return None

    def visit_Assign(self, node):
        target = node.targets[0]
        if isinstance(target, ast.Name):
            op = self.get_reduction_op(node)
            self.reduction_ops.setdefault(target.id, set()).add(op)
        return node


class MarkReductionStmts(ast.NodeTransformer):
    def __init__(self, reduction_ops):
        self.reduction_ops = reduction_ops
        self.reduction_vars = {}

    def visit_Assign(self, node):
        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id in self.reduction_ops and len(self.reduction_ops[target.id]) == 1:
                reduce_op = next(iter(self.reduction_ops[target.id]))
                if reduce_op is not None:
                    self.reduction_vars[target.id] = reduce_op
                    node.reduce = reduce_op
        return node


class MarkReductionStmtsInLoops(ast.NodeTransformer):
    def visit_For(self, node):
        self.generic_visit(node)
    
        if hasattr(node, 'pragma_dict'):
            # Perform reduction analysis: if all assignments to a scalar variable
            # only ever had one type of reduction ops, mark it as a reduction. 
            # For example, if it has both None and +, then it is not a reduction.
            visitor = GetAssignReductionOps()
            visitor.visit(node)
            reduction_ops = visitor.reduction_ops

            visitor = MarkReductionStmts(reduction_ops)
            node = visitor.visit(node)

            pragma_dict = node.pragma_dict
            reduction_vars = visitor.reduction_vars
            if len(reduction_vars) > 0 and ('parallel_for' in pragma_dict or 'simd' in pragma_dict):
                entries = []
                if 'reduction' in pragma_dict:
                    entries = pragma_dict['reduction'].split(',')

                for var,op in reduction_vars.items():
                    entry = f'{op}:{var}'
                    if entry not in entries:
                        entries.append(entry)

                pragma_dict['reduction'] = ','.join(entries)
        return node

def transform(tree):
    return MarkReductionStmtsInLoops().visit(tree)