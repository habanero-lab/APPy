import os
import textwrap
import ast_comments as ast
from slap.ast_utils import dump, get_arg_names

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

    def get_constexpr_annotated_args(self):
        newargs = []
        for i, a in enumerate(self.arg_names):
            if self.arg_types[i] == int:
                newargs.append(a+': tl.constexpr')
            else:
                newargs.append(a)
        return newargs

    def codegen(self):
        print(dump(self.func))
        self.launcher_code += textwrap.dedent(f'''
            def kernel({', '.join(self.arg_names)}):
        ''')

        for node in self.func.body:
            if type(node) is ast.For and type(node.body[0]) == ast.Comment:
                self.gen_parallel_for(node, 1)

        self.launcher_code += textwrap.dedent(f'''
                _kernel[(blockDimx,)]({', '.join(self.arg_names)})
        ''')
        

        annotated_args = self.get_constexpr_annotated_args()
        print(annotated_args)
        self.kernel_code = textwrap.dedent(f'''
            @triton.jit
            def _kernel({', '.join(annotated_args)}):
                i = tl.program_id(0) * BLOCK
                _t0 = i + tl.arange(0, BLOCK)
                _t1 = tl.load(a+_t0)
                _t2 = tl.load(b+_t0)
                tl.store(c+_t0, _t1+_t2)
        ''')

        sample = textwrap.dedent(f'''
            {self.include_code}

            {self.kernel_code}

            {self.launcher_code}
        ''')
        return sample

    def gen_parallel_for(self, node, depth):
        range_args = [x for x in node.iter.args]
        print(range_args)
        self.launcher_code += textwrap.indent(textwrap.dedent(f'''
                blockDimx = (N+BLOCK-1) // BLOCK
        '''), '    '*depth)

        if type(node.body[0]) is ast.For:
            assert len(node.body) == 1
            self.gen_parallel_for(self, node.body[0], depth)
        
        
    def gen_parallel_reduction(self, node, depth):
        return []
        
        
