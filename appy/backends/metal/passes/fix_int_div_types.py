import ast

class FixFloatDivTypes(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            a, b = node.left, node.right
            if a.metal_type in ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint'] and \
                b.metal_type in ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint']:
                node.left = ast.Call(func=ast.Name(id='float', ctx=ast.Load()), args=[a], keywords=[])
                node.right = ast.Call(func=ast.Name(id='float', ctx=ast.Load()), args=[b], keywords=[])
                node.left.metal_type = 'float'
                node.right.metal_type = 'float'
        return node
    
    # def visit_Assign(self, node):
    #     self.generic_visit(node)
    #     if node.value.metal_type == 'float':
    #         for target in node.targets:
    #             target.metal_type = 'float'
    #     return node
    

def transform(tree):
    """
    This pass detects division between two ints, and inserts a cast to float, e.g.
    a / b is rewritten to float(a) / float(b). The return type is also updated to float.
    """
    return FixFloatDivTypes().visit(tree)