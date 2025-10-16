import os
import ast
import textwrap as tw
from pathlib import Path
import ast_transforms as at
from ..base import Backend

class TritonBackend(Backend):
    def codegen(self, loop_source, metadata):
        tree = ast.parse(loop_source).body[0]
        used_names = at.get_used_names(tree, no_funcname=True)
        # Remove loop target name from used_names since it should be a local var regardless
        used_names = [x for x in used_names if x != tree.target.id]
        
        vec_add = tw.dedent('''
        for i in appy.prange(a.shape[0]):
            c[i] = a[i] + b[i]
        ''').strip()

        sample_kernels = {
            vec_add: "vec_add"
        }

        if loop_source in sample_kernels:
            m = Path(f"{os.environ["HOME"]}/projects/APPy/appy/backends/triton/sample_kernels/{sample_kernels[loop_source]}.py").read_text()
            return m
        else:
            raise NotImplementedError()