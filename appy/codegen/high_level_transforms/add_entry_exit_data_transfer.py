'''
This transformation pass takes a function as input and 
adds entry and exit data transfer statements based the JIT compilation options.
'''
import ast
from appy.ast_utils import *

class InsertBeforeReturn(ast.NodeTransformer):
    def __init__(self, stmts):
        self.stmts = stmts

    def visit_Return(self, node):
        return self.stmts + [node]


class AddEntryExitDataTransfer(ast.NodeTransformer):
    def __init__(self, options):
        self.options = options

    def visit_FunctionDef(self, node):
        if self.options.get('entry_to_device', None):
            vars = self.options.get('entry_to_device').split(',')
            stmts = [
                to_ast_node(f'{var} = torch.tensor(np.array({var}), device="cuda")') for var in vars
            ]
            node.body = stmts + node.body

        if self.options.get('exit_to_host', None):
            vars = self.options.get('exit_to_host').split(',')
            stmts = [
                to_ast_node(f'{var} = {var}.cpu().numpy()') for var in vars
            ]
            # Insert the statements before every return statement
            node = InsertBeforeReturn(stmts).visit(node)

        return node
    

def transform(tree, options):
    return AddEntryExitDataTransfer(options).visit(tree)