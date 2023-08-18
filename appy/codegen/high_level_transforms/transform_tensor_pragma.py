
from ast import unparse
from appy.ast_utils import *
import random
import re
import copy
import ast_comments as ast

random.seed(0)

def slice_to_tuple(s):
    low, up = s.replace(' ', '').split(':')
    if low == '':
        low = 0
    if up == '':
        assert False, 'upper bound of the slice must be specified: ' + s
    return low, up

def get_init_val_for_reduction(r):
    if r == 'sum':
        return 0
    elif r == 'max':
        return float('-inf')
    elif r == 'min':
        return float('inf')
    assert False

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
                low_off = simplify(low) - simplify(key_low)
                # N is a special function in sympy and need to be replaced
                up_off = simplify(up.replace('N', 'n')) - simplify(key_up.replace('N', 'n'))
                if (low_off == up_off):
                    var = self.map[key]
                    off = int(low_off)
                    #print(low, up, key_low, key_up, 'symbolic offset:', off)
                    return new_add_node(
                        new_const_node(off),
                        new_name_node(var)
                    )

            assert False, f'slice not found in map: {(low, up)}' 

    def visit_Subscript(self, node):
        if unparse(node.slice) in ['(:, None)', '(None, :)']:
            self.generic_visit(node.value)
        else:        
            self.generic_visit(node)
        return node
        

class RewriteTensorOperation(ast.NodeTransformer):
    def __init__(self, options, arg_types):
        self.verbose = True
        self.options = options
        self.arg_types = arg_types
        self.tmp_var_count = 0

    def new_variable_name(self):
        name = f'_top_var_{self.tmp_var_count}'
        self.tmp_var_count += 1
        return name

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
        
        
        if self.options.get('auto_block'):
            default_block = 'APPY_BLOCK'
            if d[slice_to_tuple(key)]['block'] == 1:
                if self.verbose:
                    print(f'update the `block` property for slice {key} to `{default_block}`')
                d[slice_to_tuple(key)]['block'] = f'{default_block}'
                self.options.setdefault('tune', {})
                self.options['tune'][f'{default_block}'] = (1024, 512, 256, 128)

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

                # Insert either only a vidx statement or a loop + an vidx statement
                # into the parent body. Note that the inner most compute statement
                # is only added later once all dimensions are visited
                # Whether to generate a loop depends on if the property specifies no loop
                # or not (via in_reg)
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

                    # Before inserting the loop, need to consider if need to initialize 
                    # the value for reduction, if there is reduction for this dimension
                    # Furthermore, reduction has two situations:
                    # 1. targets[0] is array index (store to global memory)
                    # 2. targets[0] is a variable (store to register)
                    # Currently strategy is for 1) use pre_hook in triton configs and 
                    # for 2) support only scalar reduction variable, which we know how to 
                    # initialize. Multi-dimensional array reduction must be stored to
                    # global memory.
                    reduction_prelogue = None      
                    reduction_epilogue = None      
                    if properties['reduce']:
                        reduce_op, reduce_tensor = properties['reduce'].split(':')
                        assert isinstance(node.targets[0], (ast.Name,ast.Subscript))
                        if isinstance(node.targets[0], ast.Subscript):
                            self.options.setdefault('init_hook', [])                        
                            self.options['init_hook'].append(reduce_tensor)
                            # Doesn't change the target when storing to global memory, just update the 
                            # operator to do accumulation                            
                            newtarget = node.targets[0]
                        else:
                            # This is to gen code for patterns like a = sum(A[i,:j]), where :j is blocked in a loop
                            # Multiple possible codegen exists. Currently we initialize the original target to
                            # proper initial value. This requires dtype known.
                            import appy.codegen.typesys as typesys
                            dtype  = None
                            for e in self.arg_types.values():
                                if isinstance(e, typesys.Tensor):
                                    dtype = e.get_tl_dtype()
                                    break
                            assert dtype != None
                            # reduce_tmp = self.new_variable_name()
                            # init_value = get_init_val_for_reduction(reduce_op)
                            target_s = unparse(node.targets[0])
                            init_value = get_init_val_for_reduction(reduce_op)
                            reduction_prelogue = f'{target_s} = float("{init_value}"); {target_s} = {target_s}.to({dtype})'
                            # reduction_epilogue = f'{reduce_tensor} = tl.{reduce_op}({reduce_tmp})'                            
                            newtarget = node.targets[0]
                            
                            # print(reduction_prelogue)
                            # print(reduction_epilogue)
                            # exit(1)

                        newtarget_s = unparse(newtarget)
                        if reduce_op == 'sum':                            
                            newnode_s = f'{newtarget_s} = {newtarget_s} + {unparse(node.value)}'                    
                        elif reduce_op == 'max':                            
                            newnode_s = f'{newtarget_s} = tl.maximum({newtarget_s}, {unparse(node.value)})'                                                        
                        elif reduce_op == 'min':                            
                            newnode_s = f'{newtarget_s} = tl.minimum({newtarget_s}, {unparse(node.value)})'                                                        
                        else:
                            assert False, 'unsupported'
                        node = to_ast_node(newnode_s)

                    if reduction_prelogue:
                        parent.body += to_ast_nodes(reduction_prelogue)

                    if properties['parallel']:
                        parent.body.append(ast.Comment(value='#pragma parallel'))
                    parent.body.append(loop)

                    if reduction_epilogue:
                        parent.body.append(to_ast_node(reduction_epilogue))
                    parent = loop

                    
                
            # Add statement to innermost loop (now `parent` points to)
            parent.body.append(RewriteSlice(slice_to_var).visit(node))            
            return module.body
        else:
            return node
