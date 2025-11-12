import os
import ast
import textwrap as tw
from pathlib import Path


def codegen(loop_source, loop_name, val_map, options):
    vec_add = ast.parse(tw.dedent('''
    for i in appy.prange(a_shape_0):
        c[i] = a[i] + b[i]
    ''').strip())

    if ast.dump(ast.parse(loop_source)) == ast.dump(vec_add):
        m = Path(f"{os.environ["HOME"]}/projects/APPy/appy/backends/cuda/sample_kernels/vec_add.py").read_text()
        ns = {}
        exec(m, ns)
        return ns[loop_name]
    else:
        raise NotImplementedError()
    