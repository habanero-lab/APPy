import ast

def is_name_or_constant_indexing(node):
    return isinstance(node, ast.Name) or isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)

def to_str(node):
    return ast.unparse(node)


class RewriteToAtomic(ast.NodeTransformer):
    def __init__(self, var_to_op):
        self.var_to_op = var_to_op

    def visit_Assign(self, node):
        target_str = ast.unparse(node.targets[0])
        if target_str in self.var_to_op:
            reduce_op = self.var_to_op[target_str]
            if reduce_op == '+':
                assert isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add)
                node = ast.Expr(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='tl', ctx=ast.Load()),
                            attr='atomic_add'
                        ),
                        args=[node.value.left, node.value.right]
                    )
                )
            elif reduce_op == 'max':
                assert isinstance(node.value, ast.Call) and node.value.func.id == 'max'
                node = ast.Expr(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='tl', ctx=ast.Load()),
                            attr='atomic_max'
                        ),
                        args=[node.value.args[0], node.value.args[1]]
                    )
                )
            elif reduce_op == 'min':
                assert isinstance(node.value, ast.Call) and node.value.func.id == 'min'
                node = ast.Expr(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='tl', ctx=ast.Load()),
                            attr='atomic_min'
                        ),
                        args=[node.value.args[0], node.value.args[1]],
                        keywords=[]
                    )
                )
            else:
                assert False, f'Unsupported reduction operator: {reduce_op}'

        return node
    

class FixAtomicStoreAddress(ast.NodeTransformer):
    '''
    This pass fixes code like tl.atomic_add(a[0], val) to tl.atomic_add(a+0, val).
    '''
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and ast.unparse(node.func).startswith('tl.atomic_'):
            if isinstance(node.args[0], ast.Subscript):
                node.args[0] = ast.BinOp(
                    op=ast.Add(),
                    left=node.args[0].value,
                    right=node.args[0].slice
                )
        return node


class LowerAtomicReduction(ast.NodeTransformer):
    def visit_For(self, node):
        self.generic_visit(node)
        if hasattr(node, 'pragma'):
            pragma = node.pragma
            if 'reduction' in pragma and 'parallel_for' in pragma:
                items = pragma['reduction'].split(',')
                var_to_op = {item.split(':')[1]: item.split(':')[0] for item in items}
              
                node = RewriteToAtomic(var_to_op).visit(node)
                node = FixAtomicStoreAddress().visit(node)
        return node
    
def transform(node):
    return LowerAtomicReduction().visit(node)