import os
import ast
import textwrap as tw
from pathlib import Path
from ..base import Backend

class CUDABackend(Backend):
    def codegen(self, tree, metadata):
        vec_add = ast.parse(tw.dedent('''
        for i in appy.prange(a_shape_0):
            c[i] = a[i] + b[i]
        ''').strip())

        if ast.dump(tree) == ast.dump(vec_add):
            m = Path(f"{os.environ["HOME"]}/projects/APPy/appy/backends/cuda/sample_kernels/vec_add.py").read_text()
            return m
        else:
            raise NotImplementedError()
        
    def exec(self, str, val_map):        
        exec(str, val_map)