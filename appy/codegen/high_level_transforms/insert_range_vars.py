from appy.ast_utils import *

class RangeArgsToVarsTransformer(ast.NodeTransformer):
    '''
    This pass rewrites range(...) to range(__range_var). The created new vars
    will be added a dict.
    '''
    def __init__(self):
        self.new_vars = {} # map variable name to its AST value

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        # If the function name is range, and its arguments are not names or constants, replace
        # them with new variables.
        if isinstance(node.func, ast.Name) and node.func.id == 'range':
            new_args = []
            for arg in node.args:
                if not isinstance(arg, (ast.Name, ast.Constant)):
                    new_var = f'__range_var{len(self.new_vars)}'
                    self.new_vars[new_var] = arg
                    new_args.append(new_name_node(new_var))
                else:
                    new_args.append(arg)
            node.args = new_args
        return node
        

class InsertRangeVar(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if hasattr(node, 'pragma_dict') and 'parallel_for' in node.pragma_dict:
            range_args_to_vars = RangeArgsToVarsTransformer()
            node = range_args_to_vars.visit(node)
            new_vars = range_args_to_vars.new_vars
            new_stmts = [new_assign_node(new_name_node(var), value) for var, value in new_vars.items()]
            # Insert the new variables before the loop
            return new_stmts + [node]
        return node


def transform(node):
    return InsertRangeVar().visit(node)