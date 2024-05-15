from ast import unparse
from appy.ast_utils import *
from .utils import *

class InsertRangeVar(ast.NodeTransformer):
    def __init__(self) -> None:
        self.var_count = 0

    def get_new_var(self):
        new_var = f'__rewrite_for_range_var{self.var_count}'
        self.var_count += 1
        return new_var
        
    def visit_For(self, node: ast.For):
        new_stmts = []
        if isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
            new_args = []
            for arg in node.iter.args:
                if not isinstance(arg, (ast.Name, ast.Constant)):
                    new_var = self.get_new_var()
                    new_args.append(new_name_node(new_var))                
                    new_stmts.append(
                        new_assign_node(
                            new_name_node(new_var, ctx=ast.Store()),
                            arg,
                            lineno=node.lineno
                        )
                    )                
                else:
                    new_args.append(arg)
            node.iter.args = new_args
        return new_stmts, node

