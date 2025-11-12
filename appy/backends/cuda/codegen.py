import os
import ast
import textwrap as tw
from pathlib import Path


def codegen(loop_source, loop_name, val_map, options):
    vec_add_v0 = tw.dedent('''
    for i in appy.prange(a_shape_0):
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

    if loop_source.strip() in kernel_map:
        m = Path(f"{os.environ["HOME"]}/projects/APPy/appy/backends/cuda/sample_kernels/{kernel_map[loop_source.strip()]}").read_text()
        if options.get("dump_code"):
            print(f"--- Dumped code for loop {loop_name} ---")
            print(m)
            print(f"--- End of dumped code for loop {loop_name} ---")
        ns = {}
        exec(m, ns)
        return ns[loop_name]
    else:
        raise NotImplementedError()
    