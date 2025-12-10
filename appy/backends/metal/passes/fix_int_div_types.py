import ast

class FixFloatDivTypes(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            a, b = node.left, node.right
            if a.appy_type == 'int' and b.appy_type == 'int':
                node.left = ast.Call(func=ast.Name(id='float', ctx=ast.Load()), args=[a], keywords=[])
                node.right = ast.Call(func=ast.Name(id='float', ctx=ast.Load()), args=[b], keywords=[])
                node.left.appy_type = 'float'
                node.right.appy_type = 'float'
                node.appy_type = 'float'
        return node
    
    # def visit_Assign(self, node):
    #     self.generic_visit(node)
    #     if node.value.appy_type == 'float':
    #         for target in node.targets:
    #             target.appy_type = 'float'
    #     return node
    

def transform(tree):
    """
    This pass detects division between two ints, and inserts a cast to float, e.g.
    a / b is rewritten to float(a) / float(b). The return type is also updated to float.
    """

    return FixFloatDivTypes().visit(tree)