from ast import unparse
from appy.ast_utils import *
from .utils import *
from copy import deepcopy

class ReplaceScalars(ast.NodeTransformer):
    def __init__(self, scalars):
        self.scalars = scalars

    def visit_Name(self, node):
        if node.id in self.scalars:
            node = ast.Subscript(
                value=ast.Name(id=node.id, ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=node.ctx
            )
        return node


class ProcessDataPragma(ast.NodeTransformer):
    def __init__(self):
        pass

    def visit_For(self, node: ast.For):    
        self.generic_visit(node)
        if hasattr(node, 'pragma_dict'):
            d = node.pragma_dict
            if d.get('global', None):
                global_scalars = d['global']
                global_scalars = [global_scalars] if type(global_scalars) == str else global_scalars
                to_gpu_stmts = []
                for scalar in global_scalars:
                    to_gpu_stmts.append(
                        to_ast_node(f'{scalar} = torch.tensor([{scalar}], device="cuda")')
                    )
                from_gpu_stmts = []
                for scalar in global_scalars:
                    from_gpu_stmts.append(
                        to_ast_node(f'{scalar} = {scalar}.cpu().item()')
                    )
                # Inside the for loop, replace all occurences of the global scalar `x` with `x[0]`
                ReplaceScalars(global_scalars).visit(node)
                return to_gpu_stmts + [node] + from_gpu_stmts
        
        return node
