from ast import unparse
from appy.ast_utils import *
from .utils import *
from copy import deepcopy
import appy.config as config

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
            to_device_stmts = []
            from_device_stmts = []
            if d.get('shared', None):                
                for scalar in d['shared']:
                    to_device_stmts += (
                        to_ast_node(f'{scalar} = torch.tensor(np.array({scalar}), device="cuda")'),
                    )
                    from_device_stmts += (
                        to_ast_node(f'{scalar} = {scalar}.cpu()'),
                        # Make this work for both scalar and arrays, despite the variable is called scalar
                        to_ast_node(f'{scalar} = {scalar}.item() if {scalar}.ndim == 0 else {scalar}')
                    )
                # Inside the for loop, replace all occurences of the shared scalar `x` with `x[0]`
                ReplaceScalars(d['shared']).visit(node)
            
            if d.get('to', None):
                for arr in d['to']:
                    to_device_stmts.append(
                        to_ast_node(f'{arr} = torch.from_numpy({arr}).to("cuda")')
                    )

            if d.get('from', None):
                for arr in d['from']:
                    from_device_stmts.append(
                        to_ast_node(f'{arr} = {arr}.cpu().numpy()')
                    )

            return to_device_stmts + [node] + from_device_stmts
        
        return node
