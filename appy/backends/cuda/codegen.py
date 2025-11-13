import os
import ast
import textwrap as tw
from pathlib import Path


def codegen(tree, loop_name, val_map, options):
    vec_add_v0 = tw.dedent('''
    for i in range(0, a_shape_0, 1, simd=True, parallel=True):
        c[i] = a[i] + b[i]
    ''').strip()

    vec_add_v1 = tw.dedent('''
    #pragma parallel for simd
    for i in range(a_shape_0):
        c[i] = a[i] + b[i]
    ''').strip()

    kernel_map = {
        vec_add_v0: "vec_add.py",
        vec_add_v1: "vec_add.py"
    }

    print(ast.unparse(tree))
    if ast.unparse(tree) in kernel_map:
        m = Path(f"{os.environ["HOME"]}/projects/APPy/appy/backends/cuda/sample_kernels/{kernel_map[ast.unparse(tree)]}").read_text()
        return m
    else:
        raise NotImplementedError()
    