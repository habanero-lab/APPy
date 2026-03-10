import ast
import types as _types
import ast_comments as astc
from ...utils import load_module_from_str

code_cache = {}


def _has_simd_inner_loops(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and hasattr(node, 'pragma') and node.pragma.get('simd'):
            return True
    return False


def codegen(loop_source, loop_name, val_map, options):
    kernel_val_map = {k: v for k, v in val_map.items() if not isinstance(v, _types.ModuleType)}
    tree = astc.parse(loop_source)
    types = tuple([type(v) for v in kernel_val_map.values()])
    dtypes = tuple([v.dtype if hasattr(v, 'dtype') else type(v) for v in kernel_val_map.values()])
    shapes = tuple([v.shape[1:] if hasattr(v, 'shape') else None for v in kernel_val_map.values()])
    cache_key = (loop_source, types, dtypes, shapes)

    if not options.get('clear_cache', False) and cache_key in code_cache:
        f, code_src = code_cache[cache_key]
    else:
        from ...frontend import rewrite_aug_assign
        from .passes import rewrite_tuple_assign
        from ...frontend import attach_pragma
        from .passes import lower_array_op_to_loop
        from .passes import rewrite_nested_prange
        from .passes import attach_types
        from .passes import fix_int_div_types
        from .passes import gen_host_code
        from .passes import gen_device_code

        tree = rewrite_tuple_assign.transform(tree)
        tree = rewrite_aug_assign.transform(tree)

        attach_pragma.visit(tree)
        tree = lower_array_op_to_loop.transform(tree, val_map)
        use_simd = _has_simd_inner_loops(tree)
        tree = rewrite_nested_prange.transform(tree)
        attach_types.visit(tree, val_map)
        tree = fix_int_div_types.transform(tree)

        metadata = {'loop_name': loop_name, 'val_map': kernel_val_map, 'use_simd': use_simd}
        tree, replaced_loop = gen_host_code.transform(tree, metadata)
        tree = gen_device_code.transform(tree, replaced_loop, metadata)

        ast.fix_missing_locations(tree)
        code_src = astc.unparse(tree)

        if options.get('dump_code'):
            print(f"--- Dumped code for loop {loop_name} ---")
            print(code_src.replace("\\n", "\n"))
            print(f"--- End of dumped code for loop {loop_name} ---")

        m = load_module_from_str(code_src)
        f = getattr(m, loop_name)
        code_cache[cache_key] = f, code_src

    return f, code_src


def _ensure_nvcc_in_path():
    import os
    if not any(
        os.path.isfile(os.path.join(d, 'nvcc'))
        for d in os.environ.get('PATH', '').split(os.pathsep)
    ):
        for candidate in ['/usr/local/cuda/bin', '/usr/local/cuda-12.6/bin',
                          '/usr/local/cuda-12.4/bin', '/usr/local/cuda-12/bin']:
            if os.path.isfile(os.path.join(candidate, 'nvcc')):
                os.environ['PATH'] = candidate + os.pathsep + os.environ.get('PATH', '')
                break


def exec(f, val_map):
    import numpy as np
    _ensure_nvcc_in_path()
    import pycuda.autoinit  # noqa: F401 — initializes the CUDA context
    import pycuda.gpuarray as gpuarray

    kernel_val_map = {k: v for k, v in val_map.items() if not isinstance(v, _types.ModuleType)}

    # Auto-migrate numpy arrays to GPU; track them so we can copy results back.
    # Torch CUDA tensors are wrapped zero-copy via their device pointer.
    migrated = []
    args = []
    for k, v in kernel_val_map.items():
        if type(v).__name__ == 'ndarray':
            gpu_arr = gpuarray.to_gpu(v)
            migrated.append((gpu_arr, v))
            args.append(gpu_arr)
        elif type(v).__name__ == 'Tensor' and hasattr(v, 'data_ptr') and v.is_cuda:
            import numpy as np
            dtype = np.dtype(str(v.dtype).replace('torch.', ''))
            gpu_arr = gpuarray.GPUArray(v.shape, dtype, gpudata=v.data_ptr())
            args.append(gpu_arr)
        else:
            args.append(v)

    f(*args)

    # Copy results back to original CPU arrays.
    for gpu_arr, original_arr in migrated:
        gpu_arr.get(original_arr)
