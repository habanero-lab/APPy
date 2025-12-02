import ast

class RewriteRange(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'range':
            if len(node.args) == 1:
                up = node.args[0]
                node.args = [
                    ast.Constant(0),
                    up,
                    ast.Constant(1)
                ]
            elif len(node.args) == 2:
                low, up = node.args
                node.args = [
                    low,
                    up,
                    ast.Constant(1)
                ]
            else:
                # Raise an exception if the step is not 1
                step = node.args[2]
                if not (isinstance(step, ast.Constant) and step.value == 1):
                    raise Exception(f"Only step 1 is supported in range, got: {ast.unparse(node)}")

            assert len(node.args) == 3, ast.dump(node)
        return node
    
def transform(node):
    return RewriteRange().visit(node)