import ast
from ast import unparse
from appy.ast_utils import *
import random
import re

class TritonKernelTransformer(ast.NodeTransformer):
    def __init__(self, grid):
        self.grid = grid
        self.block_dim = 0
        self.vindices = {}
        self.range_bound = {}

    def get_params_from_loop(node: ast.For):
        target = node.target
        low, up, step = node.iter.args
        return target, low, up, step

    def visit_For(self, node):
        if hasattr(node, 'pragma') and 'parallel' in node.pragma:
            index_var = node.target
            low, up, step = node.iter.args
            self.grid.append(f'(({unparse(up)} - ({unparse(low)}) + {unparse(step)} - 1) // ({unparse(step)}))')
            pid_call = new_attr_call_node('tl.program_id', [new_const_node(self.block_dim)])
            pid_stmt = new_assign_node(
                index_var, 
                new_add_node(
                    low,
                    new_mul_node(pid_call, step)
                )
            )
            
            self.block_dim += 1
            # If loop is parallel, return only its body (no explicit loop)
            self.generic_visit(node)
            node.body.insert(0, pid_stmt)
            ast.fix_missing_locations(node)            

            return node.body
        else:
            self.generic_visit(node)
            return node

    def gen_subscript_offset_deprecated(self, subscript: ast.Subscript):
        assert isinstance(subscript, ast.Subscript)
        slice = subscript.slice
        tensor = subscript.value.id
        
        #assert slice.__class__.__name__ in ('Tuple', 'Slice', 'Name', 'Constant')
        #if isinstance(slice, ast.Tuple):  # Strangely this does not work
        elts = []
        if slice.__class__.__name__ == 'Tuple':
            elts = slice.elts
        else:
            elts = [slice]
        import numpy as np
        is_elt_slice = [0] * len(elts) # np.zeros(len(elts))
        terms = []
        strides = []
        masks = []
        for i,e in enumerate(elts):
            assert type(e) in [ast.Name, ast.Slice, ast.Constant, ast.BinOp], 'unsupported slicing type: ' + unparse(e)
            
            # A bit hacky, to make indexings like `A[i, 1 + _t1]` work
            additional_offset = '0'
            if isinstance(e, ast.BinOp):
                assert isinstance(e.op, ast.Add)                
                assert isinstance(e.right, ast.Name) and (
                                            isinstance(e.left, (ast.Constant, ast.Name)) or \
                                            isinstance(e.left, (ast.UnaryOp)) and isinstance(e.left.operand, (ast.Constant, ast.Name))
                                            )
                
                if isinstance(e.right, ast.Name):                
                    additional_offset = unparse(e.left)
                    e = e.right

                # if isinstance(e.left, ast.Constant):
                #     additional_offset = str(e.left.value)
                #     e = e.right
                # elif isinstance(e.right, ast.Constant):
                #     additional_offset = str(e.right.value)
                #     e = e.left
                # else:
                #     assert False, 'unsupported slicing type: ' + unparse(e)   
                
            is_range_var = False
            if isinstance(e, ast.Name) and e.id in self.vindices:
                is_range_var = True
            

            if isinstance(e, ast.Slice) or is_range_var:
                is_elt_slice[i] = 1
            
            offset, mask = None, None
            if is_range_var:
                start, step, bound = self.vindices[e.id]
                offset = f'({unparse(start)} + tl.arange(0, {unparse(step)}))'
                if bound:
                    mask = f'({offset}) < ({unparse(bound)})'
            else:
                offset = ast.unparse(e)

            term_str = offset
            if additional_offset != '0':
                term_str = f'{additional_offset} + {term_str}'
            stride_str = f'{tensor}_stride_{i}'
            if i == len(elts) - 1:
                stride_str = '1'
            terms.append(term_str)
            strides.append(stride_str)
            masks.append(mask)
                
        # TODO: to refactor
        # If there are more than 1 slice in elements, broadcast is needed
        assert sum(is_elt_slice) in [0, 1, 2], sum(is_elt_slice)
        if sum(is_elt_slice) == 2:
            bcasts = ('[:,None]', '[None,:]')
            nonzero_indices = [i for i in range(len(is_elt_slice)) if is_elt_slice[i]]
            for i, bcast in zip(nonzero_indices, bcasts):
                terms[i] = f'({terms[i]})' + bcast

        masks = list(filter(lambda x: x!=None, masks))
        # TODO: to refactor
        if len(masks) == 0:
            mask = None
        elif len(masks) == 1:
            mask = masks[0]
        elif len(masks) == 2:
            mask = f'({masks[0]})[:,None] & ({masks[1]})[None,:]'            
        else:
            assert False
        
        strided_terms = []
        for term, stride in zip(terms, strides):
            strided_terms.append(f'({term}) * {stride}')
        offset = ' + '.join(strided_terms)
        return to_ast_expr(offset), mask


    def visit_Subscript_deprecated(self, node: ast.Subscript):
        '''
        Visit a subscript and return a `tl.load` or `tl.store` depending on the ctx.

        Currently support the following array addressing (`offset` can be either a name of constant):
            * A[constant int]
            * A[name (scalar index)]
            * A[name (vector index)]
            * A[offset + name (scalar index)]
            * A[offset + name (vector index)]
        Or a tuple of the above.
        '''
        if unparse(node.slice) in ['(:, None)', '(None, :)']:
            if isinstance(node.value, ast.Subscript):
                node.value = self.visit_Subscript(node.value)
            return node
        #dump(node)
        if '.shape[' in unparse(node):
            print('"for .. in range(x.shape[0])" is not yet supported, please pass only variable names to loop bounds, e.g. "range(N)"')
            exit(1)

        offset, mask = self.gen_subscript_offset_deprecated(node)
        
        base = node.value
        if isinstance(node.ctx, ast.Load):
            if mask == None:
                return to_ast_expr(f'tl.load({base.id} + {unparse(offset)}, mask={mask})')
            else:
                return to_ast_expr(f'tl.load({base.id} + {unparse(offset)}, mask={mask}, other=0)')
        elif isinstance(node.ctx, ast.Store):
            return to_ast_expr(f'tl.store({base.id} + {unparse(offset)}, mask={mask})')
        else:
            assert False

    def visit_Subscript(self, node: ast.Subscript):
        '''
        Visit a subscript and return a `tl.load` or `tl.store` depending on the ctx.
        '''
        if unparse(node.slice) in ['(:, None)', '(None, :)']:
            if isinstance(node.value, ast.Subscript):
                node.value = self.visit_Subscript(node.value)
            return node

        if '.shape[' in unparse(node):
            print('"for .. in range(x.shape[0])" is not yet supported, please pass only variable names to loop bounds, e.g. "range(N)"')
            exit(1)

        self.range_bound = {}  # clear the map before visiting the slices/indices
        self.generic_visit(node)  
        assert len(self.range_bound.items()) <= 1, '2d slicing is not supported'
        base = node.value
        if isinstance(node.ctx, ast.Load):
            tl_call = 'tl.load'
        elif isinstance(node.ctx, ast.Store):
            tl_call = 'tl.store' 
        else:
            assert False
        # Example: A[i], A[0]    
        if not isinstance(node.slice, ast.Tuple):
            offset = node.slice
            mask = None
            if offset in self.range_bound:
                mask = to_ast_expr(f'{unparse(offset)} < {unparse(self.range_bound[offset])}')
            addr = new_add_node(base, offset)
            
        else:
            # Example: A[i, j], A[0, i]
            addr = base
            mask = None
            for i,offset in enumerate(node.slice.elts):
                if offset in self.range_bound:
                    mask = to_ast_expr(f'{unparse(offset)} < {unparse(self.range_bound[offset])}')
                addr = new_add_node(addr, new_mul_node(offset, new_name_node(f'{base.id}_stride_{i}')))

        if mask:
            newnode = new_attr_call_node(
                tl_call, 
                [addr],
                keywords={'mask': mask}
            )
        else:
            newnode = new_attr_call_node(
                tl_call,
                [addr], 
            )
        #print(unparse(newnode))
        return newnode

    def visit_Name(self, node: ast.Name):
        if node.id not in self.vindices:
            return node

        # Process range variables
        start, step, bound = self.vindices[node.id]
        offset = to_ast_expr(f'({unparse(start)} + tl.arange(0, {unparse(step)}))')
        if bound:
            self.range_bound[offset] = bound
        return offset

    def visit_Assign(self, node: ast.Assign):
        #ast.NodeTransformer.generic_visit(self, node)
        
        lhs = node.targets[0]
        if is_call(node.value, ['vidx', 'vindex']) or is_attr_call(node.value, 'appy.vidx'):
            assert isinstance(lhs, ast.Name)
            start, stepsize = node.value.args[0:2]            
            bound = None
            if len(node.value.args) == 3:
                bound = node.value.args[2]
            else:                
                keywords = get_keyword_args(node.value)
                if 'bound' in keywords:
                    bound = keywords['bound']
            self.vindices[lhs.id] = (start, stepsize, bound)

            return None

        # This will modify `node`
        self.generic_visit(node)

        # dump(node)
        # if hasattr(node, 'pragma'):
        #     print(node.pragma)
        #     print(unparse(node))

        # Update if node is a store
        lhs = node.targets[0]
        is_atomic = False
        if hasattr(node, 'pragma') and 'atomic' in node.pragma:
            is_atomic = True

        if unparse(lhs).startswith('tl.store'):
            lhs: ast.Attribute = lhs
            value_to_be_stored = node.value
            if is_atomic:                                
                if is_add(node.value):
                    #print(unparse(node.value.left))
                    #print(unparse(lhs))
                    assert(unparse(node.value.left).replace(', other=0', '') == unparse(lhs).replace('store', 'load'))
                    value_to_be_stored = node.value.right
                    lhs.func.attr = 'atomic_add'
                    
                elif is_call(node, ('max', 'min')):
                    assert(unparse(node.value.args[0]) == unparse(lhs).replace('store', 'load'))
                    value_to_be_stored = node.value.args[1]
                    lhs.func.attr = 'atomic_' + node.value.func.id

            # Insert the value to be stored into arguments
            lhs.args.insert(1, value_to_be_stored)
            newnode = ast.Expr(value=lhs)            
            return newnode
        else:            
            return node
        
    def visit_BinOp(self, node: ast.BinOp):        
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            node = new_attr_call_node(
                'tl.dot', 
                [node.left, node.right], 
                keywords={'allow_tf32': to_ast_expr('False')}
            )
        return node