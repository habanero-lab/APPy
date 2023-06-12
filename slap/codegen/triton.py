import os
import textwrap
import ast_comments as ast
from slap.ast_utils import dump

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
        newbody = []
        for child in self.func.body:
            need_replace = False
            if type(child) is ast.For and type(child.body[0]) == ast.Comment:
                comment = child.body[0].value
                if comment.startswith('#pragma'):
                    need_replace = True
                    if comment == '#pragma parallel':
                        newstmts = self.gen_parallel_for(child)
                    elif comment.startswith('#pragma parallel reduction'):
                        newstmts = self.gen_parallel_reduction(child)

            if need_replace:
                for s in newstmts:
                    newbody.append(s)
            else:
                newbody.append(child)
        self.func.body = newbody
        
        self.launcher_code = textwrap.dedent('''
            def kernel(a, b, c, N, BLOCK):
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
        
        
