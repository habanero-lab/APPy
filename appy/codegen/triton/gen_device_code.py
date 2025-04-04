import ast
from ast import unparse
from appy.ast_utils import *

class TritonKernelTransformer(ast.NodeTransformer):
    def __init__(self, grid):
        self.grid = grid
        self.block_dim = 0
        self.vindices = {}

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
            self.vindices = {}   # vindices are per loop based
            self.generic_visit(node)
            return node

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

        self.generic_visit(node)  
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
            addr = new_add_node(base, offset)            
        else:
            # Example: A[i, j], A[0, i]
            addr = base
            for i,offset in enumerate(node.slice.elts):                
                addr = new_add_node(addr, new_mul_node(offset, new_name_node(f'{base.id}_stride_{i}')))

        if hasattr(node, 'mask'):
            newnode = new_attr_call_node(
                tl_call, 
                [addr],
                keywords={'mask': to_ast_expr(node.mask)}
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
        if isinstance(node.ctx, ast.Load):
            start, step, bound = self.vindices[node.id]
            offset = to_ast_expr(f'({unparse(start)} + tl.arange(0, {unparse(step)}))')
            return offset
        else:
            return node
        
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        # If the arg of `tl.sum` is masked, convert it to `tl.sum(tl.where(arg.mask, arg, 0))`
        if unparse(node).startswith('tl.sum('):
            if hasattr(node.args[0], 'mask'):
                arg = node.args[0]
                node.args[0] = to_ast_expr(f'tl.where({arg.mask}, {unparse(arg)}, 0)')
                #print(f'converted to: {unparse(node)}')
        return node

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
            #node.value = to_ast_expr(f'{unparse(start)} + tl.arange(0, {unparse(stepsize)})')

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