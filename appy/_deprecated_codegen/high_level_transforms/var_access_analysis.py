import ast

class VariableAccessAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.info = {}

    def set_default(self, name):
        if name not in self.info:
            self.info[name] = (set(), set(), set()) # (types, ndims, read/write flags)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name):
            ndim = len(node.slice.elts) if isinstance(node.slice, ast.Tuple) else 1
            name = node.value.id
            self.set_default(name)
            self.info[name][0].add('tensor')
            self.info[name][1].add(ndim)
            if isinstance(node.ctx, ast.Load):
                self.info[name][2].add('load')
            elif isinstance(node.ctx, ast.Store):
                self.info[name][2].add('store')
        else:
            self.generic_visit(node)

    def visit_Call(self, node):
        # Exclude function names, but visit the arguments
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node):
        name = node.id
        self.set_default(name)
        self.info[name][0].add('scalar')
        self.info[name][1].add(0)
        if isinstance(node.ctx, ast.Load):
            self.info[name][2].add('load')
        elif isinstance(node.ctx, ast.Store):
            self.info[name][2].add('store')
        
    def visit_Attribute(self, node):
        return node


def visit(tree):
    analyzer = VariableAccessAnalyzer()
    analyzer.visit(tree)
    return analyzer.info