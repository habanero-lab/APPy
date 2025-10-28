import ast

class AddBuiltinImports(ast.NodeTransformer):
    '''
    A pass that adds necessary built-in imports to the AST.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    def __init__(self):
        self.builtin_funcs = set()

    def visit_Module(self, node):
        self.generic_visit(node)
        # Add imports for built-in functions used
        # For example, if 'prange' is used, add 'from appy import prange'
        import_stmts = []
        for f in self.builtin_funcs:
            import_stmt = ast.ImportFrom(
                module='appy',
                names=[ast.alias(name=f, asname=None)],
                level=0
            )
            import_stmts.append(import_stmt)
        node.body = import_stmts + node.body
        ast.fix_missing_locations(node)
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in {'prange'} or node.func.id.startswith("ptx_ld_param_"):
            self.builtin_funcs.add(node.func.id)
        self.generic_visit(node)
        return node