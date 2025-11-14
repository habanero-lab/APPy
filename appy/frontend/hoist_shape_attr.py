import ast


class RewriteShapeAttrAccess(ast.NodeTransformer):
    def __init__(self):
        self.shape_accesses = []
        
    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Attribute) and node.value.attr == 'shape' \
              and isinstance(node.slice, ast.Constant) and isinstance(node.value.value, ast.Name):
            array_name = node.value.value.id
            dim_index = node.slice.value
            self.shape_accesses.append((array_name, dim_index))
            return ast.Name(id=f"{array_name}_shape_{dim_index}", ctx=ast.Load())
        return node


class HoistShapeAttrAccessForLoop(ast.NodeTransformer):
    def visit_For(self, node):
        visitor = RewriteShapeAttrAccess()
        visitor.visit(node)
        shape_accesses = visitor.shape_accesses
        assigns = [
            ast.Assign(
                targets=[ast.Name(id=f"{array_name}_shape_{dim_index}", ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Attribute(
                        value=ast.Name(id=array_name, ctx=ast.Load()),
                        attr='shape',
                        ctx=ast.Load()
                    ),
                    slice=ast.Constant(value=dim_index),
                    ctx=ast.Load()
                ),
                lineno=node.lineno
            )
            for array_name, dim_index in shape_accesses
        ]
        return assigns + [node]


def transform(tree):
    '''
    Transforms the AST by hoisting shape attribute accesses.
    '''    
    tree = HoistShapeAttrAccessForLoop().visit(tree)
    return tree