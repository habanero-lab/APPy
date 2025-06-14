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


class ReplaceNameWithSubscript(ast.NodeTransformer):
    def __init__(self, scalars):
        self.scalars = scalars

    def visit_Name(self, node):
        if node.id in self.scalars:
            node = ast.Subscript(
                value=ast.Name(id=node.id, ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=node.ctx
            )
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
                new_stmts = []
                for reduction in d.get('reduction').split(','):
                    assert len(reduction.split(':')) == 2
                    op, scalars = reduction.split(':')
                    assert op in ['+'], f'Unsupported reduction op: {op}'
                    # `scalars` can contain multiple variables separated by comma
                    scalars = scalars.split(',')

                    # Attach atomic pragma
                    node = AttachAtomicPragma(op, scalars).visit(node)
                    # # Add the scalar to shared clause
                    # if 'shared' in d:
                    #     d['shared'] += scalars
                    # else:
                    #     d['shared'] = scalars

                    # Add the scalars to "to" and "from" clause
                    d['to'] = d.get('to', []) + scalars
                    d['from'] = d.get('from', []) + scalars

                    # Rewrite a scalar reference to a subscript with slice 0
                    node = ReplaceNameWithSubscript(scalars).visit(node)
                    # Reference process_data_pragma.py for the compiler-generated statements
                    new_stmts.extend([to_ast_node(f'{var} = __ttc_{var}.item()') for var in scalars])
                return node, new_stmts
        return node
                

def transform(tree):
    return ProcessReductionPragma().visit(tree)