import ast


class FixIntDivTypes(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        int_types = {'bool', 'int8_t', 'uint8_t', 'int16_t', 'uint16_t', 'int', 'unsigned int'}
        if isinstance(node.op, ast.Div):
            a, b = node.left, node.right
            if getattr(a, 'cuda_type', None) in int_types and \
                    getattr(b, 'cuda_type', None) in int_types:
                node.left = ast.Call(
                    func=ast.Name(id='float', ctx=ast.Load()), args=[a], keywords=[])
                node.right = ast.Call(
                    func=ast.Name(id='float', ctx=ast.Load()), args=[b], keywords=[])
                node.left.cuda_type = 'float'
                node.right.cuda_type = 'float'
        return node


def transform(tree):
    '''
    Detects integer division and inserts float casts so Python semantics
    (int / int => float) are preserved in the generated CUDA C code.
    '''
    return FixIntDivTypes().visit(tree)
