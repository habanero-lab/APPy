from ast import unparse
from appy.ast_utils import *
from .utils import *
from copy import deepcopy

class AttachAtomicPragma(ast.NodeTransformer):
    def __init__(self, op, scalars):
        self.op = op
        self.scalars = scalars

    def visit_Assign(self, node):
        '''
        For the "+" reduction op, we attach the atomic pragma to statements
        like "x = x + y"
        '''
        rhs = node.value
        lhs = node.targets[0]
        if isinstance(lhs, ast.Name) and lhs.id in self.scalars:
            if self.op == '+' and isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Add):
                if isinstance(rhs.left, ast.Name) and lhs.id == rhs.left.id:
                    node.pragma = '#pragma atomic'
                elif isinstance(rhs.right, ast.Name) and lhs.id == rhs.right.id:
                    node.pragma = '#pragma atomic'

        return node


class ProcessReductionPragma(ast.NodeTransformer):
    '''
    This pass converts a loop annotated with reduction clause into a loop
    using explicit #pragma atomic and the shared clause.
    '''
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if hasattr(node, 'pragma_dict'):
            d = node.pragma_dict
            # If reduction is used together with parallel for, convert the reduction
            # variables to shared scope, and attach atomic pragma
            if d.get('reduction', None) and d.get('parallel_for', None):
                assert len(d['reduction'].split(':')) == 2
                op, scalars = d['reduction'].split(':')
                assert op in ['+'], f'Unsupported reduction op: {op}'
                # `scalars` can contain multiple variables separated by comma
                scalars = scalars.split(',')
                node = AttachAtomicPragma(op, scalars).visit(node)
                # Add the scalar to shared clause
                if 'shared' in d:
                    d['shared'] += scalars
                else:
                    d['shared'] = scalars

        return node
                