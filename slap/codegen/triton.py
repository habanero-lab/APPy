import os
import textwrap
import ast_comments as ast
from slap.ast_utils import dump, get_arg_names, get_first_noncomment_child

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

    def gen_launcher_node(self, node):
        if self.is_parallel_for(node):
            annotated_args = self.get_constexpr_annotated_args()
            kf = ast.parse(textwrap.dedent(f'''
                @triton.jit
                def _kernel({', '.join(annotated_args)}):
                    pass
            ''')).body[0]
            # TODO: need to save `self.kf` if not None, to support multiple kernels 
            self.kf = kf
            self.gen_parallel_for(node)

            grid = f'({",".join(self.usedBlockDims)},)'
            self.append_stmts(self.lf, f'_kernel[{grid}]({",".join(self.arg_names)})')
        else:
            stmts = ast.unparse(node)
            self.append_stmts(self.lf, stmts)
        

    def gen_kernel_node(self, node):
        stmts = ''
        if isinstance(node, ast.Assign):
            stmts = self.gen_assign(node)
        elif isinstance(node, ast.Subscript):
            stmts = self.gen_subscript(node)
        elif isinstance(node, ast.BinOp):
            stmts = self.gen_binOp(node)
        elif isinstance(node, ast.Constant):
            stmts = node.value
        elif isinstance(node, ast.Name):
            stmts = node.id
        else:
            print('pass gen_kernel_node: ')
            dump(node)
            if not isinstance(node, ast.Comment):
                assert False
        return stmts

    # def gen_kernel_node(self, node):
    #     if isinstance(node, ast.Assign):
    #         stmts = self.gen_assign(node)
    #     elif isinstance(node, ast.Subscript):
    #         stmts = self.gen_subscript(node)
    #     return stmts

    def gen_assign(self, node):
        left = node.targets[0]
        right = node.value
        rightstr = ''
        if isinstance(right, ast.Call):
            rightstr = self.gen_call(right)
            
        elif isinstance(right, ast.BinOp):
            rightstr = self.gen_binOp(right)
    
        else:
            assert False, ast.dump(node)

        stmt = ''
        if isinstance(left, ast.Name):
            stmt = f'{left.id} = {rightstr}'
        elif isinstance(left, ast.Subscript):
            # Store operation
            stmt = f'tl.store({self.gen_subscript(left)}, {rightstr})'

        return stmt


    def gen_binOp(self, node):
        left = self.gen_kernel_node(node.left)
        right = self.gen_kernel_node(node.right)
        op = self.gen_op(node.op)
        return f'{left} {op} {right}'

    def gen_slice(self, node: ast.Slice):
        low = node.lower
        up = node.upper
        assert isinstance(up, ast.BinOp)
        return f'{self.gen_kernel_node(low)} + tl.arange(0, {self.gen_kernel_node(up.right)})'

    def gen_subscript(self, node: ast.Subscript, value=None):
        dump(node)
        tensor = node.value.id
        if isinstance(node.slice, ast.Name):
            slice = node.slice.id
        elif isinstance(node.slice, ast.Slice):
            slice = self.gen_slice(node.slice)
        else:
            assert False

        if isinstance(node.ctx, ast.Load):    
            varname = f'_t{self.var_count}'
            self.append_stmts(self.kf, f'{varname} = tl.load({tensor}+{slice})')
            self.var_count += 1
            return varname
        elif isinstance(node.ctx, ast.Store):
            return f'{tensor}+{slice}'
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
        if node.func.id == 'range':
            start = node.args[0].id
            if isinstance(node.args[1], ast.BinOp):
                step = node.args[1].right.id
                return f'{start} + tl.arange(0, {step})'
            else:
                assert False, 'range must be in the form `range(i,i+BLOCK)`'

    def codegen(self):
        lf = ast.parse(textwrap.dedent(f'''
            def kernel({', '.join(self.arg_names)}):
                pass
        ''')).body[0]

        

        self.lf = lf
        

        for node in self.func.body:
            self.gen_launcher_node(node)
            

        #self.append_stmts(lf, 'blockDimx = (N+BLOCK-1) // BLOCK')
        


        # kf_body = textwrap.dedent(f'''
        #     _t1 = tl.load(a+ii)
        #     _t2 = tl.load(b+ii)
        #     tl.store(c+ii, _t1+_t2)
        # ''')

        # self.append_stmts(kf, kf_body)

        m = ast.parse(textwrap.dedent('''
            import triton
            import triton.language as tl
        '''
        ))
        m.body += [self.kf, self.lf]
        return ast.unparse(m)

    def gen_parallel_for(self, node):
        range_args = [x for x in node.iter.args]
        print(range_args)
        start, end, step = '0', '', '1'
        if len(range_args) == 1:
            end = range_args[0].id
        elif len(range_args) == 2:
            start = range_args[0].value
            end = range_args[1].id
        elif len(range_args) == 3:
            start = range_args[0].value
            end = range_args[1].id
            step = range_args[2].id

        blockDim = self.allBlockDims.pop(0)
        if step != '1':
            self.append_stmts(self.lf, f'blockDim_{blockDim} = ({end}+{step}-1) // {step}')
        else:
            self.append_stmts(self.lf, f'blockDim_{blockDim} = {end}')
        self.usedBlockDims.append(f'blockDim_{blockDim}')

        self.append_stmts(self.kf, 'i = tl.program_id(0) * BLOCK')

        for child in node.body:
            stmts = self.gen_kernel_node(child)
            self.append_stmts(self.kf, stmts)
        return ''
        
        
    def gen_parallel_reduction(self, node, depth):
        return []
        
        
