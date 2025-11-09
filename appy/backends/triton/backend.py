import os
import ast
import textwrap as tw
from pathlib import Path
import ast_transforms as at
from ..base import Backend

class TritonBackend(Backend):
    def __init__(self, val_map=None):
        self.val_map = val_map or {}   

    def codegen(self, tree, metadata):
        vec_add = ast.parse(tw.dedent('''
        for i in appy.prange(a_shape_0):
            c[i] = a[i] + b[i]
        ''').strip())

        if ast.dump(tree) == ast.dump(vec_add):
            m = Path(f"{os.environ["HOME"]}/projects/APPy/appy/backends/triton/sample_kernels/vec_add.py").read_text()
            return m
        else:
            raise NotImplementedError()