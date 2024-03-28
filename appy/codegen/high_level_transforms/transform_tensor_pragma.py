
from ast import unparse
from appy.ast_utils import *
import random
import re
import copy
import ast_comments as ast
from .utils import *

random.seed(0)

def get_init_val_for_reduction(r):
    if r == 'sum':
        return 0.0
    elif r == 'max':
        return float('-inf')
    elif r == 'min':
        return float('inf')
    assert False

class RewriteSlice(ast.NodeTransformer):
    def __init__(self, map):
        self.map = map

    def visit_Slice(self, node):
        #print('visit Slice', unparse(node), hasattr(node, 'flip'))
        low, up = slice_to_tuple(unparse(node))
        if (low, up) in self.map:
            var = self.map[(low, up)]
            if not hasattr(node, 'flip'):
                return new_name_node(var)
            else:
                # Example
                # flip(0:N) => N-1::-1 
                # flip(1:N) => N-1:0:-1
                #print('found a flip:', unparse(node))
                assert int(low) == 0, f'Only support flip on slices starting with 0'
                return to_ast_expr(f'-{var}+{up}-1')
        else:
            assert not hasattr(node, 'flip'), f'unsupported: {unparse(node)}'
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
    def __init__(self, options, arg_val_map):
        self.verbose = False
        self.options = options
        self.arg_val_map = arg_val_map
        self.tmp_var_count = 0

    def new_variable_name(self):
        name = f'_top_var_{self.tmp_var_count}'
        self.tmp_var_count += 1
        return name

    def make_prelogue_epilogue(self, reduce_op, target):
        target_s = unparse(target)
        init_value = get_init_val_for_reduction(reduce_op)
        if isinstance(target, ast.Subscript):            
            acc_var = self.new_variable_name()
            prelogue = [
                to_ast_node(f'{target_s} = float("{init_value}")'),
                to_ast_node(f'{acc_var} = {target_s}')
                #to_ast_node(f'{acc_var} = tl.zeros([BM, BN], dtype=tl.float32)')
            ]
            # Synchronization optimization
            prelogue[0].no_sync = True
            epilogue = [
                to_ast_node(f'{target_s} = {acc_var}')
            ]
            target = to_ast_expr(acc_var)
        else:
            #import appy.codegen.typesys as typesys
            import torch
            dtype  = None
            for e in self.arg_val_map.values():

                if isinstance(e, torch.Tensor):
                    dtype = 'tl.' + str(e.dtype).replace('torch.', '')
                    break
            assert dtype != None, 'failed to get data type of the operation'

            prelogue = [
                #to_ast_node(f'{target_s} = float("{init_value}")'),
                to_ast_node(f'{target_s} = {init_value}'),
                to_ast_node(f'{target_s} = {target_s}.to({dtype})'),
            ]
            epilogue = None
        return target, prelogue, epilogue

    def visit_Assign(self, node):        
        if hasattr(node, 'pragma') and '=>' in node.pragma:
            # Perform pre-passes before lowering to loops
            from .rewrite_call import RewriteAPPyCall
            node = RewriteAPPyCall().visit(node)

            module = ast.Module(body=[])
            parent = module
            init_hook = None            
            if hasattr(node, 'init_hook'):
                init_hook = node.init_hook
            
            slice_to_var = {}
            pragma = node.pragma
            slice_map = parse_pragma(pragma)
            #print(slice_map)
            if self.verbose:
                print(slice_map)
                
            if self.options.get('auto_block') or self.options.get('auto_simd'):
                default_block = 'APPY_BLOCK'
                last_slice = list(slice_map.keys())[-1]
                if slice_map[last_slice]['block'] == 1:
                    if self.verbose:
                        print(f'update the `block` property for slice {last_slice} to `{default_block}`')

                    if slice_map[last_slice]['reduce']:
                        slice_map[last_slice]['block'] = f'1024'
                    else:                    
                        slice_map[last_slice]['block'] = f'128'

                    #slice_map[last_slice]['block'] = f'{default_block}'
                    #self.options.setdefault('tune', {})
                    #self.options['tune'][f'{default_block}'] = (1024, 512, 256, 128)

            for slice,properties in slice_map.items():
                low, up = slice
                step = properties['block']
                index_var = self.new_variable_name()                
                
                slice_to_var[(low,up)] = index_var

                # Insert either only a vidx statement or a loop + an vidx statement
                # into the parent body. Note that the inner most compute statement
                # is only added later once all dimensions are visited
                # Whether to generate a loop depends on if the property specifies no loop
                # or not (via single_block)
                if properties['single_block']:
                    vidx_stmt = new_assign_node(
                                new_name_node(index_var),
                                new_call_node(
                                    'vidx', [
                                        to_ast_expr(str(low)),
                                        to_ast_expr(str(step)),
                                        to_ast_expr(str(up)),                                        
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
                        to_ast_expr(str(low)),
                        to_ast_expr(str(up)),
                        to_ast_expr(str(step)),
                    )
                    loop.from_tensor_expr = True
                    
                    # Make index vectorized if step size is > 1
                    # `index_var` is reused
                    if step != 1:
                    #if True:
                        vidx_stmt = new_assign_node(
                                new_name_node(index_var),
                                new_call_node(
                                    'vidx', [
                                        new_name_node(index_var),
                                        to_ast_expr(str(step)),
                                        to_ast_expr(str(up)),
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
                    loop_prelogue = None      
                    loop_epilogue = None      
                    if properties['reduce']:
                        if ':' in properties['reduce']:
                            reduce_op, reduce_tensor = properties['reduce'].split(':')
                        else:
                            reduce_op = properties['reduce']
                        assert isinstance(node.targets[0], (ast.Name,ast.Subscript))
                        # if isinstance(node.targets[0], ast.Subscript):
                        #     #self.options.setdefault('init_hook', [])                        
                        #     #self.options['init_hook'].append(reduce_tensor)
                        #     #self.options['init_hook'] = [reduce_tensor]
                        #     #loop.init_hook = reduce_tensor
                        #     # Doesn't change the target when storing to global memory, just update the 
                        #     # operator to do accumulation                            
                        #     newtarget = node.targets[0]
                        # else:
                            
                        # This is to gen code for patterns like a = sum(A[i,:j]), where :j is blocked in a loop
                        # Multiple possible codegen exists. Currently we initialize the original target to
                        # proper initial value. This requires dtype known.
                        # import appy.codegen.typesys as typesys
                        # dtype  = None
                        # for e in self.arg_val_map.values():
                        #     if isinstance(e, typesys.Tensor):
                        #         dtype = e.get_tl_dtype()
                        #         break
                        # assert dtype != None
                        # # reduce_tmp = self.new_variable_name()
                        # # init_value = get_init_val_for_reduction(reduce_op)
                        # target_s = unparse(node.targets[0])
                        # init_value = get_init_val_for_reduction(reduce_op)
                        # loop_prelogue = f'{target_s} = float("{init_value}"); {target_s} = {target_s}.to({dtype})'
                        # if isinstance(node.targets[0], ast.Subscript):
                        #     loop_prelogue = f'{target_s} = float("{init_value}")'
                        # # reduction_epilogue = f'{reduce_tensor} = tl.{reduce_op}({reduce_tmp})'                            
                        # newtarget = node.targets[0]
                        
                        # # print(loop_prelogue)
                        # # print(reduction_epilogue)
                        # # exit(1)
                        new_target, loop_prelogue, loop_epilogue = self.make_prelogue_epilogue(reduce_op, node.targets[0])
                        newtarget_s = unparse(new_target)
                        if reduce_op == 'sum':                            
                            newnode_s = f'{newtarget_s} = {newtarget_s} + {unparse(node.value)}'                    
                        elif reduce_op == 'max':                            
                            newnode_s = f'{newtarget_s} = tl.maximum({newtarget_s}, {unparse(node.value)})'                                                        
                        elif reduce_op == 'min':                            
                            newnode_s = f'{newtarget_s} = tl.minimum({newtarget_s}, {unparse(node.value)})'                                                        
                        else:
                            assert False, 'unsupported'
                        node = to_ast_node(newnode_s)


                    if loop_prelogue:
                        for stmt in loop_prelogue:
                            parent.body.append(RewriteSlice(slice_to_var).visit(stmt))                        

                    if properties['parallel']:
                        parent.body.append(ast.Comment(value='#pragma parallel'))
                    parent.body.append(loop)

                    if loop_epilogue:
                        for stmt in loop_epilogue:
                            #dump(stmt)
                            parent.body.append(RewriteSlice(slice_to_var).visit(stmt))  
                    parent = loop

                                    
            # Add statement to innermost loop (now `parent` points to)
            parent.body.append(RewriteSlice(slice_to_var).visit(node)) 
            if init_hook:          
                for child in module.body:
                    if isinstance(child, ast.For):
                        child.init_hook = init_hook
                        break
            return module.body
        else:
            return node
