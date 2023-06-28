import os
import re
import torch
import textwrap
from copy import deepcopy
import ast_comments as ast
from ast import unparse
from .typesys import build_type_from_value, get_tl_dtype_from_str
from .typesys import Tensor as TensorType
from .typesys import Constant as ConstantType
from ..ast_utils import dump, dump_code, get_arg_names, new_call_node, to_ast_node, to_ast_expr

class TritonBackend(object):
    def __init__(self, ast_tree, arg_values):
        for node in ast_tree.body:
            if isinstance(node, ast.FunctionDef):
                self.func = node
                break
        self.arg_values = list(arg_values)
        for keyword_arg in self.func.args.defaults:
            assert isinstance(keyword_arg, ast.Constant)
            self.arg_values.append(keyword_arg.value)
        
        
        self.module = ast.parse(textwrap.dedent('''
            import torch
            import triton
            import triton.language as tl
        '''
        ))
        self.arg_names = get_arg_names(self.func)
        self.arg_types = [build_type_from_value(x) for x in arg_values]
        self.allBlockDims = ['x', 'y', 'z']
        self.usedBlockDims = []
        self.kernel_count = 0
        self.var_count = 0
        self.reduction_vars = []
        self.lf_local_vars = {}
        self.range_vars = []
        self.range_var_mask = {}
        self.index_block_sizes = {}
        self.index_bounds = {}
        self.var_types = {}

    def get_arg_value(self, arg_name):
        for name, value in zip(self.arg_names, self.arg_values):
            if name == arg_name:
                return value
        assert False, f"argument `{arg_name}` not found"

    def get_constexpr_annotated_args(self):
        newargs = []
        for i, a in enumerate(self.arg_names):
            if isinstance(self.arg_types[i], ConstantType):
                newargs.append(a+': tl.constexpr')
            else:
                newargs.append(a)
        return newargs

    def append_stmts(self, parent, stmts):
        n = ast.parse(stmts).body
        parent.body += n

    def is_parallel_for(self, node):
        return type(node) is ast.For and type(node.body[0]) == ast.Comment and '#pragma parallel' in node.body[0].value

    def is_sequential_for(self, node):
        return type(node) is ast.For and (type(node.body[0]) != ast.Comment or '#pragma parallel' not in node.body[0].value)

    def get_kernel_function_parameters(self):
        newargs = []
        for name, val in zip(self.arg_names, self.arg_values):
            if type(val) in (int, torch.dtype):
                newargs.append(name+': tl.constexpr')
            elif type(val) == torch.Tensor:
                if val.layout != torch.strided:
                    continue
                newargs.append(name)
                for d in range(val.dim()):
                    newargs.append(f'{name}_stride_{d}: tl.constexpr')
            else:
                newargs.append(name)

        for var, is_const in self.lf_local_vars.items():
            if is_const:
                newargs.append(var+': tl.constexpr')
            else:
                newargs.append(var)
        #print(newargs)
        #exit(1)
        return newargs

    def get_kernel_function_arguments(self):
        newargs = []
        for name, val in zip(self.arg_names, self.arg_values):
            if type(val) == torch.Tensor:
                if val.layout != torch.strided:
                    continue
                newargs.append(name)
                for d in range(val.dim()):
                    newargs.append(f'{name}.stride({d})')
            elif type(val) == torch.dtype:
                newargs.append(str(val).replace('torch.', 'tl.'))
            else:
                newargs.append(name)

        for var, is_const in self.lf_local_vars.items():
            newargs.append(var)
        return newargs

    def gen_launcher_node(self, node):
        if self.is_parallel_for(node):
            pragma = node.body[0].value
            self.record_const_vars(pragma)
            
            kernel_name = f'_kernel{self.kernel_count}'
            self.kernel_count += 1
            k_params = self.get_kernel_function_parameters()
            k_args = self.get_kernel_function_arguments()
            kf = ast.parse(textwrap.dedent(f'''
                @triton.jit
                def {kernel_name}({', '.join(k_params)}):
                    pass
            ''')).body[0]
            
            self.kf = kf
            self.gen_parallel_for(node)

            grid = f'({",".join(self.usedBlockDims)},)'
            self.append_stmts(self.lf, f'fn = {kernel_name}[{grid}]({",".join(k_args)})')
            #self.append_stmts(self.lf, 'print(fn.asm["ptx"])')
            #self.append_stmts(self.lf, 'exit(1)')
            self.module.body.append(kf)
        else:
            if isinstance(node, ast.Assign):
                #dump(node)
                assert len(node.targets) == 1
                target = node.targets[0]
                assert isinstance(target, ast.Name) or target.__class__.__name__ == 'Tuple', ast.dump(target)
                
                vars = []
                if isinstance(target, ast.Name):
                    vars.append(target.id)
                elif target.__class__.__name__ == 'Tuple':
                    for e in target.elts:
                        vars.append(e.id)

                for var in vars:
                    self.lf_local_vars[var] = False
                    
            stmts = ast.unparse(node)
            self.append_stmts(self.lf, stmts)
        

    def gen_kernel_node(self, node):
        newnode = node
        #print('gen kernel node')
        #dump(node)
        if isinstance(node, ast.Assign):
            newnode = self.gen_assign(node)
        elif isinstance(node, ast.Subscript):
            newnode = self.gen_subscript(node)
        elif isinstance(node, ast.BinOp):
            newnode = self.gen_binOp(node)
        elif isinstance(node, ast.Constant):
            newnode = node
        elif isinstance(node, ast.Name):
            newnode = node
        elif isinstance(node, ast.AugAssign):
            newnode = self.gen_aug_assign(node)
        elif isinstance(node, ast.AnnAssign):
            newnode = self.gen_assign(node)
        elif isinstance(node, ast.Slice):
            newnode = self.gen_slice(node)
        elif isinstance(node, ast.Call):
            newnode = self.gen_call(node)
        elif isinstance(node, ast.For):
            newnode = self.gen_for(node)
        elif isinstance(node, ast.Compare):
            newnode = node
        else:
            if not isinstance(node, ast.Comment):
                assert False, ast.dump(node)
        return newnode

    def gen_for(self, node: ast.For):
        newloop = deepcopy(node)
        newbody = []
        newloop.body = newbody
        
        loop_index = node.target.id
        self.var_types[loop_index] = TensorType(torch.int32, 0)
        step = '1'
        if isinstance(node.body[0], ast.Comment) and node.body[0].value.startswith('#pragma'):
            pragma = node.body[0].value

            match = re.search(r' reduction\((.*?)\)', pragma)
            if match:
                reduction_var = match.groups()[0]
                self.reduction_vars.append(reduction_var)

            match = re.search(r' block\((.*?)\)', pragma)
            if match:
                step = match.groups()[0]
                self.index_block_sizes[loop_index] = step
                # Update range
                rangenode = to_ast_node(f'{loop_index} = range({loop_index}, {loop_index}+{step})')
                node.body.insert(1, rangenode)
                        

        for child in node.body:
            newbody.append(self.gen_kernel_node(child))
        
        
        assert newloop.iter.func.id == 'range'
        range_args = newloop.iter.args
        newargs = []
        for arg in range_args:
            newargs.append(self.gen_kernel_node(arg))

        if step != '1':
            newargs.append(self.gen_kernel_node(arg))
        newloop.iter.args = newargs

        return newloop

    # def gen_kernel_node(self, node):
    #     if isinstance(node, ast.Assign):
    #         stmts = self.gen_assign(node)
    #     elif isinstance(node, ast.Subscript):
    #         stmts = self.gen_subscript(node)
    #     return stmts

    def get_tl_dtype(self, dtype):
        return 'tl.' + dtype.replace('torch.', '')

    def block_reduction(self, node):
        newnode = ast.Assign
        newtarget = None
        newvalue = None
        if isinstance(node, ast.AugAssign):
            if (isinstance(node.target, ast.Name) and node.target.id in self.reduction_vars) or \
                (isinstance(node.target, ast.Subscript) and node.target.value.id in self.reduction_vars):    
                # Example: s += a[i]  =>  s = sum(a[i])
                # Reduction statement found
                newtarget = node.target
                if isinstance(node.op, ast.Add):
                    newvalue = ast.Call(func=ast.Name('sum'), args=[node.value], keywords=[])
                else:
                    assert False, f'non-sum reduction type unsupported yet: {ast.dump(node)}'
            
        if isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.reduction_vars:
                newtarget = node.targets[0]
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                    # Example: s = s + a[i]  =>  s = sum(a[i])
                    assert unparse(newtarget) == unparse(node.value.left)
                    newvalue = ast.Call(func=ast.Name('sum'), args=[node.value.right], keywords=[])
                elif isinstance(node.value, ast.Call) and node.value.func.id in 'max, min':
                    assert unparse(newtarget) == unparse(node.value.args[0])
                    # Example: s = max(s, a[i])  =>  s = max(a[i])
                    newvalue = ast.Call(func=ast.Name('sum'), args=[node.value.args[1]], keywords=[])

            if isinstance(node.targets[0], ast.Subscript) and node.targets[0].value.id in self.reduction_vars:
                newtarget = node.targets[0]
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                    # Example: s[0] = s[0] + a[i]  =>  s = sum(a[i])
                    assert unparse(newtarget) == unparse(node.value.left)
                    newvalue = ast.Call(func=ast.Name('sum'), args=[node.value.right], keywords=[])
                elif isinstance(node.value, ast.Call) and node.value.func.id in 'max, min':
                    # Example: s[0] = max(s[0] + a[i])  =>  s = max(a[i])
                    assert unparse(newtarget) == unparse(node.value.args[0])
                    newvalue = ast.Call(func=ast.Name('sum'), args=[node.value.args[1]], keywords=[])

        if newtarget:
            newnode = ast.Assign(targets=[newtarget], value=newvalue, lineno=node.lineno)
            dump(newnode)
            print(unparse(newnode))
            exit(1)
            return 
        else:
            return node

    def gen_assign(self, node):
        if isinstance(node, ast.Assign):
            left = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            left = node.target
        right = node.value
    
        if isinstance(left, ast.Name):
            newnode = ast.Assign(targets=[left], lineno=node.lineno)
            # In our programming model, storing to a global array must be a subscript 
            # expression, even if the array has only one element.
            
            
            if isinstance(node.value, ast.Subscript) and node.value.value.id in self.range_vars:
                # Mask pattern
                name = node.value.value.id
                dump(node)
                print(self.range_var_mask[name])
                exit(1)
                # what about statements like   a[i[i<N]]
            else:
                newright = self.gen_kernel_node(right)
                if isinstance(node.value, ast.Constant):
                    assert isinstance(self.arg_types[0], TensorType)
                    dtype = self.arg_types[0].get_tl_dtype()
                
                    if isinstance(node, ast.AnnAssign):
                        dtype = get_tl_dtype_from_str(node.annotation.id)
                
                    blocksizes = self.index_block_sizes.values()
                    if len(blocksizes) > 0:
                        e = f'tl.zeros([{",".join(blocksizes)}], dtype={dtype}) + {node.value.value}'
                        newright = to_ast_expr(e)

                if isinstance(newright, ast.Expr):
                    newright = newright.value

                if isinstance(node.value, ast.Call) and node.value.func.id in ['range', 'arange']:
                    self.range_vars.append(left.id)

                newnode.value = newright
            

            if hasattr(newright, 'type'):
                self.var_types[left.id] = newright.type
                
                #exit(1)
                
        elif isinstance(left, ast.Subscript):
            # A normal store or an atomic store
            tensor = left.value.id
            slice = unparse(self.gen_subscript_slice(left))
            store_func = 'tl.store'
            store_value = self.gen_kernel_node(right)
            if tensor in self.reduction_vars:
                if isinstance(right, ast.BinOp) and isinstance(right.op, ast.Add):
                    # Example: s[0] = s[0] + a[i]  =>  s = sum(a[i])
                    assert unparse(left) == unparse(right.left), "Example: s[0] = s[0] + a[i:i+BLOCK]"
                    store_func = 'tl.atomic_add'
                    store_value = self.gen_kernel_node(right.right)
                    
                elif isinstance(right, ast.Call) and right.func.id in ['maximum', 'minimum']:
                    # Example: s[0] = maximum(s[0] + a[i:i+BLOCK])  =>  s = maximum(a[i:i+BLOCK])
                    assert unparse(left) == unparse(right.args[0]), "Example: s[0] = maximum(s[0] + a[i:i+BLOCK])"
                    store_func = 'tl.atomic_max' if right.func.id == 'maximum' else 'tl.atomic_min'
                    store_value = self.gen_kernel_node(right.args[1])
                else:
                    assert False, "Supported reduction op: maximum, minimum and +"
                    
            s = f'{store_func}({tensor}+{slice}, {unparse(store_value)})'
            newnode = to_ast_node(s)        
        else:
            assert False
        
        return newnode

    def gen_aug_assign(self, node):
        left = node.target
        right = node.value
        newnode = ast.Assign(targets=[left], lineno=node.lineno)
        leftcopy = deepcopy(left)
        if isinstance(leftcopy, ast.Subscript):
            leftcopy.ctx = ast.Load()
        newnode.value = ast.BinOp(left=leftcopy, op=node.op, right=node.value)
        return self.gen_kernel_node(newnode)
    
        # if isinstance(left, ast.Name):
        #     newnode = ast.Assign(targets=[left], lineno=node.lineno)
        #     newnode.value = self.gen_kernel_node(right)
                
        # elif isinstance(left, ast.Subscript):
        #     # A normal store or an atomic store
        #     tensor = left.value.id
        #     slice = unparse(self.gen_subscript_slice(left))
        #     store_func = 'tl.store'
        #     store_value = self.gen_kernel_node(right)
        #     if tensor in self.reduction_vars:
        #         if isinstance(node.op, ast.Add):
        #             # Example: s[0] += a[i]  =>  s = sum(a[i])
        #             store_func = 'tl.atomic_add'
        #             store_value = self.gen_kernel_node(right)
        #         else:
        #             assert False
                    
        #     s = f'{store_func}({tensor}+{slice}, {unparse(store_value)})'
        #     newnode = to_ast_node(s)        
        # else:
        #     assert False
        
        # return newnode

    def gen_binOp(self, node):
        left = self.gen_kernel_node(node.left)
        right = self.gen_kernel_node(node.right)

        if isinstance(node.op, ast.MatMult):
            s = f'tl.dot({unparse(left)}, {unparse(right)})'
            newnode = to_ast_expr(s)
        else:
            newnode = ast.BinOp(op=node.op, left=left, right=right)
        return newnode

    def gen_slice(self, node: ast.Slice):
        low = node.lower
        up = node.upper
       
        if low is None:
            s = f'tl.arange(0, {ast.unparse(self.gen_kernel_node(up))})'
        else:
            if isinstance(up, ast.BinOp):
                assert low.id == up.left.id
                s = f'{ast.unparse(self.gen_kernel_node(low))} + tl.arange(0, {ast.unparse(self.gen_kernel_node(up.right))})'
            else:
                s = f'tl.arange({ast.unparse(self.gen_kernel_node(low))}, {ast.unparse(self.gen_kernel_node(up))})'

        return to_ast_node(s)

    def gen_subscript_slice(self, subscript: ast.Subscript):
        assert isinstance(subscript, ast.Subscript)
        slice = subscript.slice
        tensor = subscript.value.id
        
        #assert slice.__class__.__name__ in ('Tuple', 'Slice', 'Name', 'Constant')
        #if isinstance(slice, ast.Tuple):  # Strangely this does not work
        if slice.__class__.__name__ == 'Tuple':
            all_1d = True
            for i,e in enumerate(slice.elts):
                assert type(e) in [ast.Name, ast.Slice, ast.Constant]

                if isinstance(e, ast.Name):
                    ndim = 0
                    if e.id in self.var_types:
                        var_type = self.var_types[e.id]
                        assert var_type.ndim <= 1
                        ndim = var_type.ndim
                        
                    if ndim == 0:
                        all_1d = False

                if isinstance(e, ast.Constant):
                    all_1d = False

            terms = []
            shape_broadcasts = []
            ndim = len(slice.elts)
            if ndim == 1:
                shape_broadcasts.append('')
            else:
                for i in range(ndim):
                    slices = []
                    for _ in range(i):
                        slices.append('None')
                    slices.append(':')
                    for _ in range(i+1, ndim):
                        slices.append('None')
                    shape_broadcasts.append(f'[{",".join(slices)}]')
                
            for i,e in enumerate(slice.elts):
                term_str = ast.unparse(self.gen_kernel_node(e))
                if i != len(slice.elts) - 1:
                    term_str = f'({term_str}) * {tensor}_stride_{i}'
                if all_1d:
                    term_str = f'({term_str}){shape_broadcasts[i]}'
                terms.append(term_str)
                    
            slice = ' + '.join(terms)
            return to_ast_expr(slice)
        else:
            return self.gen_kernel_node(slice)


    def gen_subscript(self, node: ast.Subscript, value=None):
        if unparse(node.slice) in ['(None, :)', '(:, None)']:
            return ast.Subscript(value=self.gen_kernel_node(node.value), slice=node.slice)

        tensor = node.value.id
        slice = unparse(self.gen_subscript_slice(node))
        assert isinstance(node.ctx, ast.Load)

        if isinstance(node.ctx, ast.Load):    
            # varname = f'_t{self.var_count}'
            # self.append_stmts(self.kf, f'{varname} = tl.load({tensor}+{slice})')
            # self.var_count += 1
            # return varname
            return to_ast_expr(f'tl.load({tensor}+{slice})')
            # TODO: perhaps directly add `mask=tensor_shape_x` here, which requires knowning the dim and shapes
        # elif isinstance(node.ctx, ast.Store):
        #     if tensor in self.reduction_vars:
        #         # TODO: to add other atomic operations
        #         stmt = f'tl.atomic_add({tensor}+{slice}, {ast.unparse(value)})'
        #     else:
        #         stmt = f'tl.store({tensor}+{slice}, {ast.unparse(value)})'
        #     return to_ast_node(stmt)
        # else:
        #     assert False

    def gen_op(self, op):
        if isinstance(op, ast.Add):
            return '+'
        elif isinstance(op, ast.Sub):
            return '-'
        elif isinstance(op, ast.Mult):
            return '*'
        elif isinstance(op, ast.Div):
            return '/'
        else:
            assert False, f'unknown operator: {ast.dump(op)}'
        
    def gen_call(self, node: ast.Call):
        node_s = unparse(node)
        funcname = node.func.id
        stmt = ''
        node_type = None
        if funcname in ['range', 'arange']:
            start_s = node.args[0].id
            if isinstance(node.args[1], ast.BinOp):
                step_arg = node.args[1].right
                step_s = ast.unparse(self.gen_kernel_node(step_arg))
                stmt = f'{start_s} + tl.arange(0, {step_s})'
                if isinstance(step_arg, ast.Constant):
                    node_type = TensorType(torch.int32, 1, shape=[int(step_s)])
                elif isinstance(step_arg, ast.Name):
                    node_type = TensorType(torch.int32, 1, shape=[self.get_arg_value(step_s)])
                else:
                    assert False
            else:
                assert False, 'range must be in the form `range(i,i+BLOCK)`'
        elif funcname in ['sum', 'max', 'min']: 
            args = []
            for arg in node.args:
                args.append(ast.unparse(self.gen_kernel_node(arg)))              
            keyword_args = []
            for keyword in node.keywords:
                arg, value = keyword.arg, keyword.value.value
                args.append(f'{arg}={value}')
                keyword_args.append(arg)
            if 'axis' not in keyword_args:
                args.append(f'axis=0')
            
            stmt = f'tl.{funcname}({",".join(args)})'
        elif funcname in ['zeros', 'empty']:
            shape = node.args[0]
            if shape.__class__.__name__ in ['List', 'Tuple']:
                shape_arg = unparse(shape)
            else:
                shape_arg = f'[{unparse(shape)}]'
            
            dtype = re.search(r'dtype=(.*)\)', node_s).groups()[0].replace('torch.', 'tl.')
            stmt = f'tl.{funcname}({shape_arg}, dtype={dtype})'
        elif funcname in ['exp', 'log', 'maximum']:
            args = []
            for arg in node.args:
                args.append(ast.unparse(self.gen_kernel_node(arg))) 
            stmt = f'tl.{funcname}({",".join(args)})'
        else:
            assert False, 'unknown function call: ' + ast.dump(node)
        newnode = to_ast_expr(stmt)
        if node_type != None:
           newnode.type = node_type
        return newnode

    def record_const_vars(self, pragma):
        match = re.search(r' const\((.*?)\)', pragma)
        if match:
            const_vars = match.groups()[0]
            print(const_vars)
            for var in const_vars.split(','):
                var = var.strip()
                self.lf_local_vars[var] = True
                

    def gen_parallel_for(self, node: ast.For):
        #dump(node)
        range_args = [x for x in node.iter.args]
        #print(range_args)
        start, end, step = '0', '', '1'
        if len(range_args) == 1:
            end = ast.unparse(self.gen_kernel_node(range_args[0]))
        elif len(range_args) == 2:
            start = ast.unparse(self.gen_kernel_node(range_args[0]))
            end = ast.unparse(self.gen_kernel_node(range_args[1]))
        elif len(range_args) == 3:
            start = ast.unparse(self.gen_kernel_node(range_args[0]))
            end = ast.unparse(self.gen_kernel_node(range_args[1]))
            step = ast.unparse(self.gen_kernel_node(range_args[2]))

        loop_index = node.target.id
        self.var_types[loop_index] = TensorType(torch.int32, 0)
        pragma = node.body[0].value


        match = re.search(r' reduction\((.*?)\)', pragma)
        if match:
            reduction_var = match.groups()[0]
            self.reduction_vars.append(reduction_var)
        
        match = re.search(r' block\((.*?)\)', pragma)
        if match:
            step = match.groups()[0]
            self.index_block_sizes[loop_index] = step
            newnode = to_ast_node(f'{loop_index} = range({loop_index}, {loop_index}+{step})')
            node.body.insert(1, newnode)

            # # Need to add reduction call when blocking a reduction loop
            # if 'reduction' in pragma:
            #     for child in node.body:
            #         if isinstance(child, ast.AugAssign):
            #             right = child.value
            #             if isinstance(child.op, ast.Add):
            #                 child.value = to_ast_node(f'sum({unparse(right)})').value
            #             else:
            #                 assert False, f'reduction type unsupported yet {ast.dump(child)}'
                        

        blockDim = self.allBlockDims.pop(0)
        if step != '1':
            self.append_stmts(self.lf, f'blockDim_{blockDim} = ({end}-{start}+{step}-1) // {step}')
        else:
            self.append_stmts(self.lf, f'blockDim_{blockDim} = {end}')
    
        self.append_stmts(self.kf, f'{loop_index} = {start} + tl.program_id({len(self.usedBlockDims)}) * {step}')

        self.usedBlockDims.append(f'blockDim_{blockDim}')

        for child in node.body:
            if self.is_parallel_for(child):
                self.gen_parallel_for(child)
            else:
                newnode = self.gen_kernel_node(child)
                assert not isinstance(newnode, str), newnode
                if not isinstance(newnode, ast.Comment):
                    self.kf.body.append(newnode)
        return ''
        
    def codegen(self):
        # lf = ast.parse(textwrap.dedent(f'''
        #     def kernel({', '.join(self.arg_names)}):
        #         pass
        # ''')).body[0]
        lf = ast.FunctionDef(name='kernel', args=self.func.args, body=[], decorator_list=[], lineno=self.func.lineno)
        self.lf = lf
        
        for node in self.func.body:
            self.gen_launcher_node(node)
            
        m = self.module
        #dump(self.kf)
        m.body += [self.lf]
        return ast.unparse(m)
