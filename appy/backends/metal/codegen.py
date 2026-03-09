import os
import ast
import types as _types
import ast_comments as astc
from pathlib import Path
from ...utils import load_module_from_str

code_cache = {}

def _has_simd_inner_loops(tree):
    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and hasattr(node, 'pragma') and node.pragma.get('simd'):
            return True
    return False

def codegen(loop_source, loop_name, val_map, options):
    kernel_val_map = {k: v for k, v in val_map.items() if not isinstance(v, _types.ModuleType)}
    tree = astc.parse(loop_source)
    types = tuple([type(v) for v in kernel_val_map.values()])
    dtypes = tuple([v.dtype if hasattr(v, "dtype") else type(v) for v in kernel_val_map.values()])
    shapes = tuple([v.shape[1:] if hasattr(v, "shape") else None for v in kernel_val_map.values()])
    cache_key = (loop_source, types, dtypes, shapes)
    if options.get("clear_cache", False) == False and cache_key in code_cache:
        f, code_src = code_cache[cache_key]
    else:
        # Do frontend transformation
        from ...frontend import rewrite_aug_assign
        #from ...frontend import rewrite_tuple_assign
        from .passes import rewrite_tuple_assign

        tree = rewrite_tuple_assign.transform(tree)
        tree = rewrite_aug_assign.transform(tree)

        # Metal specific codegen
        from ...frontend import attach_pragma
        from .passes import lower_array_op_to_loop
        from .passes import rewrite_nested_prange
        from .passes import attach_types
        from .passes import fix_int_div_types
        from .passes import gen_host_code
        from .passes import gen_device_code

        attach_pragma.visit(tree)
        tree = lower_array_op_to_loop.transform(tree, val_map)
        use_simd = _has_simd_inner_loops(tree)
        tree = rewrite_nested_prange.transform(tree)
        attach_types.visit(tree, val_map)
        tree = fix_int_div_types.transform(tree)

        metadata = {'loop_name': loop_name, 'val_map': kernel_val_map, 'use_simd': use_simd}
        tree, replaced_loop = gen_host_code.transform(tree, metadata)
        tree = gen_device_code.transform(tree, replaced_loop, metadata)

        # code_src = Path(f"{Path(__file__).parent}/sample_kernels/gelu.py").read_text()
        # m = load_module_from_str(code_src)
        # f = getattr(m, loop_name)
        ast.fix_missing_locations(tree)
        code_src = astc.unparse(tree)
      
        if options['dump_code']:
            print(f"--- Dumped code for loop {loop_name} ---")          
            print(code_src.replace("\\n", "\n"))
            print(f"--- End of dumped code for loop {loop_name} ---")

        m = load_module_from_str(code_src)
        f = getattr(m, loop_name)

        code_cache[cache_key] = f, code_src

    return f, code_src


def exec(f, val_map):
    import numpy as np
    from ...np_shared import array_to_buffer, device
    val_map = {k: v for k, v in val_map.items() if not isinstance(v, _types.ModuleType)}

    # Track arrays that were auto-migrated so we can copy results back.
    migrated = []  # list of (metal_arr, original_arr)

    args = []
    for k, v in val_map.items():
        if type(v).__name__ == 'ndarray':
            if v.ctypes.data not in array_to_buffer:
                # Auto-migrate: allocate a Metal shared buffer, copy data in.
                buf = device.buffer(v.nbytes)
                metal_arr = np.frombuffer(buf, dtype=v.dtype, count=v.size).reshape(v.shape)
                metal_arr[:] = v
                array_to_buffer[metal_arr.ctypes.data] = buf
                migrated.append((metal_arr, v))
                args.append(buf)
            else:
                args.append(array_to_buffer[v.ctypes.data])
        else:
            args.append(v)

    assert device, "Could not find device"
    args.append(device)
    f(*args)

    # Copy results back to original arrays so callers see the updated values.
    for metal_arr, original_arr in migrated:
        original_arr[:] = metal_arr