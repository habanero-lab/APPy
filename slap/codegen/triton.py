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

    def gen_node(self, node, lf, kf):
        stmts = ''
        if self.is_parallel_for(node):
            stmts = self.gen_parallel_for(node, lf, kf)
        
        else:
            print('pass gen_node')
            dump(node)
            pass
        return stmts

    def gen_kernel_node(self, node, kf):
        if isinstance(node, ast.Assign):
            stmts = self.gen_assign(node, kf)
        elif isinstance(node, ast.Subscript):
            stmts = self.gen_subscript(node, kf)
        return stmts

    def gen_assign(self, node, kf):
        left = node.targets[0]
        right = node.value
        rightstr = ''
        if isinstance(right, ast.Call):
            rightstr = self.gen_call(right, kf)
            
        elif isinstance(right, ast.BinOp):
            rightstr = self.gen_binOp(right, kf)
        else:
            print('pass gen_node')
            dump(node)

        stmt = ''
        if isinstance(left, ast.Name):
            stmt = f'{left.id} = {rightstr}'
        elif isinstance(left, ast.Subscript):
            # Store operation
            stmt = f'tl.store({self.gen_subscript(left)}, {rightstr})'

        return stmt


    def gen_binOp(self, node, kf):
        dump(node)
        left = self.gen_kernel_node(node.left, kf)
        right = self.gen_kernel_node(node.right, kf)
        op = self.gen_op(node.op)
        return f'{left} {op} {right}'

    def gen_subscript(self, node: ast.Subscript, kf, value=None):
        tensor = node.value.id
        slice = node.slice.id
        if node.ctx == ast.Load:    
            varname = f'_t{self.var_count}'
            self.append_stmts(kf, f'{varname} = tl.load({tensor}+{slice})')
            self.var_count += 1
            return varname
        elif node.ctx == ast.Store:
            return f'{tensor}+{slice}'

    def gen_tensor_load(self, node: ast.Subscript, kf):
        assert isinstance(node, ast.Subscript)
        tensor = node.value.id
        slice = node.slice.id
        varname = f'_t{self.var_count}'
        self.append_stmts(kf, f'{varname} = tl.load({tensor}+{slice})')
        self.var_count += 1
        return varname

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
        
    def gen_call(self, node, kf):
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

        annotated_args = self.get_constexpr_annotated_args()
        kf = ast.parse(textwrap.dedent(f'''
            @triton.jit
            def _kernel({', '.join(annotated_args)}):
                pass
        ''')).body[0]

        for node in self.func.body:
            self.gen_node(node, lf, kf)

        #self.append_stmts(lf, 'blockDimx = (N+BLOCK-1) // BLOCK')
        grid = f'({",".join(self.usedBlockDims)},)'
        self.append_stmts(lf, f'_kernel[{grid}]({",".join(self.arg_names)})')


        kf_body = textwrap.dedent(f'''
            _t1 = tl.load(a+ii)
            _t2 = tl.load(b+ii)
            tl.store(c+ii, _t1+_t2)
        ''')

        self.append_stmts(kf, kf_body)

        m = ast.parse(textwrap.dedent('''
            import triton
            import triton.language as tl
        '''
        ))
        m.body += [kf, lf]
        return ast.unparse(m)

    def gen_parallel_for(self, node, lf, kf):
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
            self.append_stmts(lf, f'blockDim_{blockDim} = ({end}+{step}-1) // {step}')
        else:
            self.append_stmts(lf, f'blockDim_{blockDim} = {end}')
        self.usedBlockDims.append(f'blockDim_{blockDim}')

        self.append_stmts(kf, 'i = tl.program_id(0) * BLOCK')

        for child in node.body:
            stmts = self.gen_node(child, lf, kf)
            self.append_stmts(kf, stmts)
        
        
    def gen_parallel_reduction(self, node, depth):
        return []
        
        
