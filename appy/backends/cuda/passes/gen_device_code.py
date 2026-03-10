import ast
from . import type_map
from .constants import BLOCK_SIZE


def gen_headers():
    return '''\
#include <cuda.h>
#include <stdint.h>

__device__ inline unsigned int wang_hash(unsigned int x) {
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

__device__ inline float appy_random(unsigned int seed) {
    return (float)wang_hash(seed) / (float)0xffffffffu;
}

'''


def gen_func_header(loop_name, val_map):
    params = []
    for var, val in val_map.items():
        ty = type(val).__name__
        if ty == 'ndarray':
            cuda_ty = type_map.get_cuda_type(val)
            params.append(f"{cuda_ty}* {var}")
        else:
            cuda_ty = type_map.get_cuda_type(val)
            params.append(f"const {cuda_ty} {var}")
    return f"__global__ void _{loop_name}({', '.join(params)})"


def gen_var_decls(loop, val_map, use_simd=False):
    var_to_type = {}
    for node in ast.walk(loop):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id not in val_map:
                var_to_type[target.id] = getattr(target, 'cuda_type', 'auto')

    s = ""
    for var, ty in var_to_type.items():
        if ty and ty != 'auto':
            s += f"    {ty} {var};\n"

    range_args = loop.iter.args
    low = "0" if len(range_args) == 1 else ast.unparse(range_args[0])
    loop_index = loop.target.id
    up = ast.unparse(range_args[-1])
    offset = f" + {low}" if low != "0" else ""

    if use_simd:
        # Each parallel iteration gets BLOCK_SIZE threads.
        # Total threads launched = n_iters * BLOCK_SIZE.
        # tid maps to: loop_index = tid / BLOCK_SIZE, lane = tid % BLOCK_SIZE.
        s += f"    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
        s += f"    int {loop_index} = tid / {BLOCK_SIZE}{offset};\n"
        s += f"    int lane = tid % {BLOCK_SIZE};\n"
        s += f"    if ({loop_index} >= {up}) return;\n"
    else:
        s += f"    int {loop_index} = blockIdx.x * blockDim.x + threadIdx.x{offset};\n"
        s += f"    if ({loop_index} >= {up}) return;\n"
    return s


def gen_func_body(replaced_loop):
    from .device_passes import unparse_to_cpp
    s = ""
    for child in replaced_loop.body:
        s += unparse_to_cpp.visit(child)
    return s


def transform(tree, replaced_loop, metadata):
    loop_name = metadata['loop_name']
    val_map = metadata['val_map']
    use_simd = metadata.get('use_simd', False)

    from .device_passes import rewrite_func_calls
    from .device_passes import rewrite_multi_dim_indexing
    from .device_passes import fix_random_call
    from .device_passes import rewrite_simd_loops
    from .device_passes import rewrite_simd_reductions
    from .device_passes import guard_scalar_mem_writes

    replaced_loop = rewrite_func_calls.transform(replaced_loop)
    replaced_loop = rewrite_multi_dim_indexing.transform(replaced_loop, val_map)
    replaced_loop = fix_random_call.transform(replaced_loop)
    if use_simd:
        replaced_loop = rewrite_simd_loops.transform(replaced_loop)
        replaced_loop = rewrite_simd_reductions.transform(replaced_loop)
        replaced_loop = guard_scalar_mem_writes.transform(replaced_loop)

    kernel_str = gen_headers()
    kernel_str += gen_func_header(loop_name, val_map)
    kernel_str += " {\n"
    kernel_str += gen_var_decls(replaced_loop, val_map, use_simd)
    kernel_str += gen_func_body(replaced_loop)
    kernel_str += "}\n"

    tree.body.insert(0, ast.Assign(
        targets=[ast.Name(id='kernel_str', ctx=ast.Load())],
        value=ast.Constant(value=kernel_str)
    ))
    return tree
