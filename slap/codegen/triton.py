import os
import re
import torch
import textwrap
from copy import deepcopy
import ast_comments as ast
from ast import unparse
from slap.ast_utils import dump, dump_code, get_arg_names, get_first_noncomment_child, to_ast_node

class TritonBackend(object):
    def __init__(self, ast_tree, arg_values):
        self.func = ast_tree.body[0]
        self.arg_values = arg_values
        self.launcher_code = ''
        self.kernel_code = ''
        self.include_code = textwrap.dedent('''
            import triton
            import triton.language as tl
        ''')
        self.arg_names = get_arg_names(self.func)
        self.arg_types = [type(x) for x in arg_values]
        self.allBlockDims = ['x', 'y', 'z']
        self.usedBlockDims = []
        self.var_count = 0
        self.reduction_vars = []


    def get_constexpr_annotated_args(self):
        newargs = []
        for i, a in enumerate(self.arg_names):
            if self.arg_types[i] == int:
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
            if type(val) == int:
                newargs.append(name+': tl.constexpr')
            elif type(val) == torch.Tensor:
                newargs.append(name)
                for d in range(val.dim()):
                    newargs.append(f'{name}_stride_{d}')
            else:
                newargs.append(name)
        return newargs

    def get_kernel_function_arguments(self):
        newargs = []
        for name, val in zip(self.arg_names, self.arg_values):
            if type(val) == torch.Tensor:
                newargs.append(name)
                for d in range(val.dim()):
                    newargs.append(f'{name}.stride({d})')
            else:
                newargs.append(name)
        return newargs

    def gen_launcher_node(self, node):
        if self.is_parallel_for(node):
            k_params = self.get_kernel_function_parameters()
            k_args = self.get_kernel_function_arguments()
            kf = ast.parse(textwrap.dedent(f'''
                @triton.jit
                def _kernel({', '.join(k_params)}):
                    pass
            ''')).body[0]
            # TODO: need to save `self.kf` if not None, to support multiple kernels 
            self.kf = kf
            self.gen_parallel_for(node)

            grid = f'({",".join(self.usedBlockDims)},)'
            self.append_stmts(self.lf, f'_kernel[{grid}]({",".join(k_args)})')

        else:
            stmts = ast.unparse(node)
            self.append_stmts(self.lf, stmts)
        

    def gen_kernel_node(self, node):
        newnode = node
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
            newnode = self.gen_assign(node)
        elif isinstance(node, ast.Slice):
            newnode = self.gen_slice(node)
        elif isinstance(node, ast.Call):
            newnode = self.gen_call(node)
        else:
            if not isinstance(node, ast.Comment):
                assert False, ast.dump(node)
        return newnode

    # def gen_kernel_node(self, node):
    #     if isinstance(node, ast.Assign):
    #         stmts = self.gen_assign(node)
    #     elif isinstance(node, ast.Subscript):
    #         stmts = self.gen_subscript(node)
    #     return stmts

    def gen_assign(self, node):
        if isinstance(node, ast.Assign):
            left = node.targets[0]
        elif isinstance(node, ast.AugAssign):
            left = node.target
        right = node.value

        newnode = deepcopy(node)
        if isinstance(left, ast.Name):
            newnode.targets[0] = self.gen_kernel_node(left)
            newnode.value = self.gen_kernel_node(right)
            if isinstance(newnode.value, ast.Expr):
                newnode.value = newnode.value.value
            #print('new assign node')
            #dump(newnode)
        elif isinstance(left, ast.Subscript):
            newnode = self.gen_subscript(left, value=self.gen_kernel_node(right))
            
        else:
            assert False
        return newnode


    def gen_binOp(self, node):
        left = self.gen_kernel_node(node.left)
        right = self.gen_kernel_node(node.right)
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

    def gen_subscript(self, node: ast.Subscript, value=None):
        tensor = node.value.id
        
        #if isinstance(node.slice, ast.Tuple):  # Strangely this does not work
        if node.slice.__class__.__name__ == 'Tuple':
            terms = []
            for i,e in enumerate(node.slice.elts):
                if i == len(node.slice.elts) - 1:  # stride would be 1
                    terms.append(ast.unparse(self.gen_kernel_node(e)))
                else:
                    terms.append(f'({ast.unparse(self.gen_kernel_node(e))}) * {tensor}_stride_{i}')
            slice = ' + '.join(terms)
        else:
            slice = ast.unparse(self.gen_kernel_node(node.slice))
    
        if isinstance(node.ctx, ast.Load):    
            # varname = f'_t{self.var_count}'
            # self.append_stmts(self.kf, f'{varname} = tl.load({tensor}+{slice})')
            # self.var_count += 1
            # return varname
            return to_ast_node(f'tl.load({tensor}+{slice})')
        elif isinstance(node.ctx, ast.Store):
            if tensor in self.reduction_vars:
                # TODO: to add other atomic operations
                stmt = f'tl.atomic_add({tensor}+{slice}, {ast.unparse(value)})'
            else:
                stmt = f'tl.store({tensor}+{slice}, {ast.unparse(value)})'
            return to_ast_node(stmt)
        else:
            assert False

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
        
    def gen_call(self, node):
        funcname = node.func.id
        stmt = ''
        if funcname == 'range':
            start = node.args[0].id
            if isinstance(node.args[1], ast.BinOp):
                step = ast.unparse(self.gen_kernel_node(node.args[1].right))
                stmt = f'{start} + tl.arange(0, {step})'
            else:
                assert False, 'range must be in the form `range(i,i+BLOCK)`'
        elif funcname in ['sum', 'max', 'min']:
            
            args = []
            for arg in node.args:
                args.append(ast.unparse(self.gen_kernel_node(arg)))
            #print(args)
            stmt = f'tl.{funcname}({",".join(args)}, axis=0)'
           
        else:
            assert False
        #print(stmt)
        return to_ast_node(stmt)

    def codegen(self):
        lf = ast.parse(textwrap.dedent(f'''
            def kernel({', '.join(self.arg_names)}):
                pass
        ''')).body[0]

       
        self.lf = lf
        

        for node in self.func.body:
            self.gen_launcher_node(node)
            

        m = ast.parse(textwrap.dedent('''
            import torch
            import triton
            import triton.language as tl
        '''
        ))
        #dump(self.kf)
        m.body += [self.kf, self.lf]
        return ast.unparse(m)

    def gen_parallel_for(self, node: ast.For):
        #dump(node)
        range_args = [x for x in node.iter.args]
        #print(range_args)
        start, end, step = '0', '', '1'
        if len(range_args) == 1:
            end = range_args[0].id
        elif len(range_args) == 2:
            start = range_args[0].value
            end = range_args[1].id
        elif len(range_args) == 3:
            start = ast.unparse(self.gen_kernel_node(range_args[0]))
            end = ast.unparse(self.gen_kernel_node(range_args[1]))
            step = ast.unparse(self.gen_kernel_node(range_args[2]))

        loop_index = node.target.id
        pragma = node.body[0].value


        match = re.search(r' reduction\((.*?)\)', pragma)
        if match:
            reduction_var = match.groups()[0]
            self.reduction_vars.append(reduction_var)

        match = re.search(r' block\((.*?)\)', pragma)
        if match:
            # TODO: to insert a range statement in the beginning of the loop
            step = match.groups()[0]
            newnode = to_ast_node(f'{loop_index} = range({loop_index}, {loop_index}+{step})')
            node.body.insert(1, newnode)

            if 'reduction' in pragma:
                for child in node.body:
                    if isinstance(child, ast.AugAssign):
                        right = child.value
                        if isinstance(child.op, ast.Add):
                            child.value = to_ast_node(f'sum({unparse(right)})').value
                        else:
                            assert False, f'reduction unsupported yet {ast.dump(child)}'
                        

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
        
        
    def gen_parallel_reduction(self, node, depth):
        return []
        
        
