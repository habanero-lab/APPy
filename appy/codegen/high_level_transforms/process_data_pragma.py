from ast import unparse
from appy.ast_utils import *
from .utils import *
from copy import deepcopy
import appy.config as config
    

class RenameASTNames(ast.NodeTransformer):
    def __init__(self, name_map):
        self.name_map = name_map

    def visit_Name(self, node):
        if node.id in self.name_map:
            node = ast.Name(id=self.name_map[node.id], ctx=node.ctx)
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
                        to_ast_node(f'{scalar} = torch.tensor(np.array({scalar}, copy=False), device="cuda")'),
                    )
                    from_device_stmts += (
                        to_ast_node(f'{scalar} = {scalar}.cpu()'),
                        # Make this work for both scalar and arrays, despite the variable is called scalar
                        to_ast_node(f'{scalar} = {scalar}.item() if {scalar}.ndim == 0 else {scalar}.numpy()')
                    )
            
            if d.get('to', None):
                for var in d['to']:
                    to_device_stmts.extend((
                        to_ast_node(f'__ttc_{var} = torch.from_numpy(np.array({var}, copy=False))'),
                        to_ast_node(f'__{var} = __ttc_{var}.cuda()')
                    )) 
                    RenameASTNames({var: f'__{var}'}).visit(node)

            if d.get('from', None):
                for var in d['from']:
                    from_device_stmts.extend((
                        to_ast_node(f'__ttc_{var}.copy_(__{var})'),
                        # Make this work for both scalar and arrays, despite the variable is called scalar
                        #to_ast_node(f'{var} = {var}.item() if {var}.ndim == 0 else {var}.numpy()'),
                    ))

            return to_device_stmts + [node] + from_device_stmts
        
        return node


def transform(node):
    return ProcessDataPragma().visit(node)