import os
import ast_comments as ast
from appy.ast_utils import dump

class CUDABackend(object):
    def __init__(self, ast_tree):
        self.func = ast_tree.body[0]

    def codegen(self):
        '''
        Perform code generation for function `self.ast_tree`. 
        This will generate a temporary file called '{func_name}_triton_kernel.py' in 
        directory `/tmp/appy/`.
        '''
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
        
        # Remove the @appy decorator
        self.decorator_list = []

        tmp_dir = '/tmp/appy'
        os.makedirs(f'{tmp_dir}', exist_ok=True)
        ofs = open(f'{tmp_dir}/{self.func.name}_triton_kernel.py', 'w')
        code = ast.unparse(self.func)
        print(code)
        ofs.write(code)
        ofs.close()

    def gen_parallel_for(self, node):        
        return []
        
    def gen_parallel_reduction(self, node):
        return []
        
        