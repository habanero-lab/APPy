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
        if self.is_parallel_for(node):
            self.gen_parallel_for(node, lf, kf)
        elif isinstance(node, ast.Assign):
            self.gen_assign(node, lf, kf)
        else:
            pass

    def gen_assign(self, node, lf, kf):
        pass

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
            i = tl.program_id(0) * BLOCK
            _t0 = i + tl.arange(0, BLOCK)
            _t1 = tl.load(a+_t0)
            _t2 = tl.load(b+_t0)
            tl.store(c+_t0, _t1+_t2)
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
        
        # if self.is_parallel_for(get_first_noncomment_child(node)):
        # #    assert len(node.body) == 1
        #     self.gen_parallel_for(get_first_noncomment_child(node), lf, kf)

        for child in node.body:
            self.gen_node(child, lf, kf)
        
        
    def gen_parallel_reduction(self, node, depth):
        return []
        
        
