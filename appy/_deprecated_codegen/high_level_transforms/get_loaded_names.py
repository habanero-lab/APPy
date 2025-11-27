import ast

class ExtractArguments(ast.NodeVisitor):
    def __init__(self):
        self.read_names = {}
        self.write_names = {}
        self.func_or_package_names = []

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name):            
            ndim = 1
            if isinstance(node.slice, ast.Tuple):
                ndim = len(node.slice.elts)
            self.read_names[node.value.id] = ('tensor', ndim)
            #print(self.names)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            self.func_or_package_names.append(node.func.id)
        
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            self.func_or_package_names.append(node.value.id)

        self.generic_visit(node)

    def visit_Name(self, node):   
        id = node.id     
        if id.startswith('_top_var'):# or id.startswith('__range_var'):
            return
        
        if id in self.func_or_package_names:
            return

        # `id` could already be registered as a tensor read name since
        # visit_Subscript is called before visit_Name
        if isinstance(node.ctx, ast.Load) and id not in self.read_names:
            self.read_names[id] = ('scalar', 0)

        if isinstance(node.ctx, ast.Store):
            self.write_names[id] = ('scalar', 0)        
        

def transform(tree):
    visitor = ExtractArguments()
    visitor.visit(tree)
    readonly_names = {}

    for name in visitor.read_names:
        if name not in visitor.write_names:
            readonly_names[name] = visitor.read_names[name]
            
    return tree, readonly_names