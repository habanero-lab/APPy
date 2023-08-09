import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
import random
import re
import copy

random.seed(0)

def slice_to_tuple(s):
    low, up = s.replace(' ', '').split(':')
    if low == '':
        low = 0
    if up == '':
        assert False, 'upper bound of the slice must be specified: ' + key
    return low, up

class RewriteSlice(ast.NodeTransformer):
    def __init__(self, map):
        self.map = map

    def visit_Slice(self, node):
        low, up = slice_to_tuple(unparse(node))
        if (low, up) in self.map:
            var = self.map[(low, up)]
            return new_name_node(var)
        else:
            var = None
            off = 0
            for key in self.map:
                key_low, key_up = key
                key_low, key_up = str(key_low), str(key_up)                
                from sympy import simplify
                low_off = simplify(key_low) - simplify(low)
                # N is a special function in sympy and need to be replaced
                up_off = simplify(key_up.replace('N', 'n')) - simplify(up.replace('N', 'n'))
                if (low_off == up_off):
                    var = self.map[key]
                    off = int(low_off)
                    #print('symbolic offset:', off)
                    return new_add_node(
                        new_const_node(off),
                        new_name_node(var)
                    )

            assert False, f'slice not found in map: {(low, up)}' 
        

class RewriteTensorOperation(ast.NodeTransformer):
    def __init__(self):
        self.verbose = True

    def new_variable_name(self):
        return f'_t{random.randint(0, 1e4)}'

    def parse_pragma(self, pragma):
        d = {}
        s = pragma.replace('#pragma', '')
        for item in s.split(' '):
            if item == '':
                continue
            
            key, value = item.split('=>')            
            props = {'parallel': False, 'block': 1, 'in_reg': False, 'reduce': None}
            for prop in value.split(','):
                
                match = re.search(r'\((.*?)\)', prop)
                if match:
                    arg = match.groups()[0]
                    prop_name = prop.split('(')[0]
                    props[prop_name] = arg
                else:
                    props[prop] = True
            
            d[slice_to_tuple(key)] = props
        
        return d

    def visit_Assign(self, node):        
        if hasattr(node, 'pragma'):
            module = ast.Module(body=[])
            parent = module
            
            slice_to_var = {}
            pragma = node.pragma
            slice_map = self.parse_pragma(pragma)
            if self.verbose:
                print(slice_map)

            for slice,properties in slice_map.items():
                low, up = slice
                step = properties['block']
                index_var = self.new_variable_name()                
                
                slice_to_var[(low,up)] = index_var

                if properties['in_reg']:
                    vidx_stmt = new_assign_node(
                                new_name_node(index_var),
                                new_call_node(
                                    'vidx', [
                                        new_name_node(low) if isinstance(low, str) else new_const_node(low),
                                        new_name_node(step),
                                        new_name_node(up)
                                    ]
                                )
                            )
                    parent.body.append(vidx_stmt)
                else:
                    # Generate a loop for each slice. Nested loop will be 
                    # generated if multiple slices. `parent=loop` controls this 
                    # recursive logic                    
                    loop = new_for_loop(
                        new_name_node(index_var),
                        new_name_node(low) if isinstance(low, str) else new_const_node(low), 
                        new_name_node(up),
                        new_name_node(step) if isinstance(step, str) else new_const_node(step),
                    )
                    
                    # Make index vectorized if step size is > 1
                    # `index_var` is reused
                    if step != 1:
                        vidx_stmt = new_assign_node(
                                new_name_node(index_var),
                                new_call_node(
                                    'vidx', [
                                        new_name_node(index_var),
                                        new_name_node(step),
                                        new_name_node(up)
                                    ]
                                )
                            )
                        loop.body.append(
                            vidx_stmt
                        )
                    
                    if properties['parallel']:
                        parent.body.append(ast.Comment(value='#pragma parallel'))
                    parent.body.append(loop)
                    parent = loop

                    # Rewrite the statement to do reduction
                    if properties['reduce']:
                        reduce_op = properties['reduce']
                        if reduce_op == '+':
                            target_load = copy.deepcopy(node.targets[0])
                            target_load.ctx = ast.Load()
                            node = new_assign_node(                                
                                node.targets[0],
                                new_add_node(target_load, node.value),
                                node.lineno
                            )
                            dump(node)
                            #print(unparse(node))
                            #exit(1)
                        else:
                            assert False, 'unsupported'
                
            # Add statement to innermost loop (now `parent` points to)
            parent.body.append(RewriteSlice(slice_to_var).visit(node))            
            return module.body
        else:
            return node
