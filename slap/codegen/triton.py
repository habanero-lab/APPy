import os
import textwrap
import ast_comments as ast
from slap.ast_utils import dump, get_arg_names

class TritonBackend(object):
    def __init__(self, ast_tree):
        self.func = ast_tree.body[0]
        self.launcher_code = ''
        self.kernel_code = ''
        self.include_code = textwrap.dedent('''
            import triton
            import triton.language as tl
        ''')

    def codegen(self):
        arg_names = get_arg_names(self.func)
        
        self.launcher_code = textwrap.dedent(f'''
            def kernel({', '.join(arg_names)}):
                nblocks = (N+BLOCK-1) // BLOCK
                _kernel[(nblocks,)](a, b, c, N, BLOCK)
        ''')

        self.kernel_code = textwrap.dedent('''
            @triton.jit
            def _kernel(a, b, c, N: tl.constexpr, BLOCK: tl.constexpr):
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

    def gen_parallel_for(self, node):        
        return []
        
    def gen_parallel_reduction(self, node):
        return []
        
        
