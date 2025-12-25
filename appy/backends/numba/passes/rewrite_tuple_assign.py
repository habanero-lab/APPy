import ast

class RewriteTupleAssign(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Tuple):
            if isinstance(node.targets[0], ast.Tuple):
                new_assigns = []
                for i in range(len(node.value.elts)):
                    new_assign = ast.Assign(
                        targets=[node.targets[0].elts[i]],
                        value=node.value.elts[i]
                    )
                    new_assigns.append(new_assign)
                return new_assigns
            elif isinstance(node.targets[0], ast.Subscript):
                # Add an extra index to the subscript, e.g
                # a[0] = (1, 2) is rewritten to a[0,0] = 1; a[0,1] = 2
                # The slice of the subscript may or may not be a Tuple
                new_assigns = []
                if not isinstance(node.targets[0].slice, ast.Tuple):
                    raise NotImplementedError
                
                for i in range(len(node.value.elts)):
                    new_target = ast.Subscript(
                        value=node.targets[0].value,
                        slice=ast.Tuple(elts=node.targets[0].slice.elts + [ast.Constant(value=i)]),
                        ctx=node.targets[0].ctx
                    )
                    new_assign = ast.Assign(
                        targets=[new_target],
                        value=node.value.elts[i]
                    )
                    new_assigns.append(new_assign)
                return new_assigns
        return node

def transform(node):
    '''
    This pass rewrites tuple assignments to multiple assignments, e.g.
    (a, b) = (1, 2) is rewritten to a = 1; b = 2
    '''
    return RewriteTupleAssign().visit(node)