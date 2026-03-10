import ast


class RewriteTupleAssign(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Tuple):
            if isinstance(node.targets[0], ast.Tuple):
                return [
                    ast.Assign(targets=[t], value=v)
                    for t, v in zip(node.targets[0].elts, node.value.elts)
                ]
            elif isinstance(node.targets[0], ast.Subscript):
                if not isinstance(node.targets[0].slice, ast.Tuple):
                    raise NotImplementedError
                return [
                    ast.Assign(
                        targets=[ast.Subscript(
                            value=node.targets[0].value,
                            slice=ast.Tuple(
                                elts=node.targets[0].slice.elts + [ast.Constant(value=i)]),
                            ctx=node.targets[0].ctx
                        )],
                        value=v
                    )
                    for i, v in enumerate(node.value.elts)
                ]
        return node


def transform(node):
    '''
    Rewrites tuple assignments to multiple scalar assignments, e.g.
    (a, b) = (1, 2)  =>  a = 1; b = 2
    '''
    return RewriteTupleAssign().visit(node)
