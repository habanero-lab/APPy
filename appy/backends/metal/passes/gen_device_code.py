import ast
from . import type_map
from .constants import SIMD_WIDTH

def gen_headers():
    return '''
#include <metal_stdlib>
using namespace metal;

inline uint wang_hash(uint x) {
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

inline float random(uint seed) {
    return float(wang_hash(seed)) / float(0xffffffffu);
}

'''

def gen_func_header(loop_name, replaced_loop, val_map):
    s = f"kernel void _{loop_name}("

    count = 0
    for var, val in val_map.items():
        ty = type(val).__name__
        if ty == "ndarray":
            # Arrays
            dtype = val.dtype
            s += f"device {type_map.get_metal_type(str(dtype))}* {var} [[ buffer({count}) ]], "
        else:
            # Scalars
            s += f"constant {type_map.get_metal_type(ty)}& {var} [[ buffer({count}) ]], "
        count += 1

    s += f"uint grid_id [[ thread_position_in_grid ]])"
    return s

def gen_var_decls(loop, val_map, use_simd=False):
    # Check all assigned vars, declare them if they are not in val_map
    var_to_type = {}
    for node in ast.walk(loop):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id not in val_map:
                var_to_type[target.id] = target.metal_type

    s = ""
    for var, ty in var_to_type.items():
        s += f"    {ty} {var};\n"

    range_args = loop.iter.args
    low = "0" if len(range_args) == 1 else ast.unparse(range_args[0])

    loop_index = loop.target.id
    up = ast.unparse(range_args[-1])
    offset = f" + {low}" if low != "0" else ""
    if use_simd:
        s += f"    uint {loop_index} = grid_id / {SIMD_WIDTH}{offset};\n"
        s += f"    uint lane = grid_id % {SIMD_WIDTH};\n"
    else:
        s += f"    uint {loop_index} = grid_id{offset};\n"
        # Metalcompute rounds thread count up to a multiple of the threadgroup
        # size, so guard extra threads against out-of-bounds access.
        s += f"    if ({loop_index} >= {up}) return;\n"
    return s


def gen_func_body(replaced_loop):
    from .device_passes import unparse_to_cpp
    s = ""
    for child in replaced_loop.body:
        s += unparse_to_cpp.visit(child)
    return s

#     return f'''
#     xi = x[id];
#     x3 = xi * xi * xi;
#     t = tanh(0.79788456f * (xi + 0.044715f * x3));
#     y[id] = 0.5f * xi * (1.0f + t);
# '''

def transform(tree, replaced_loop, metadata):
    loop_name = metadata['loop_name']
    val_map = metadata['val_map']
    use_simd = metadata.get('use_simd', False)

    from .device_passes import rewrite_func_calls
    from .device_passes import rewrite_multi_dim_indexing
    from .device_passes import fix_random_call
    from .device_passes import rewrite_simd_loops
    replaced_loop = rewrite_func_calls.transform(replaced_loop)
    replaced_loop = rewrite_multi_dim_indexing.transform(replaced_loop, val_map)
    replaced_loop = fix_random_call.transform(replaced_loop)
    if use_simd:
        replaced_loop = rewrite_simd_loops.transform(replaced_loop)

    kernel_str = gen_headers()
    kernel_str += gen_func_header(loop_name, replaced_loop, val_map)
    kernel_str += "{\n"
    kernel_str += gen_var_decls(replaced_loop, val_map, use_simd)
    kernel_str += gen_func_body(replaced_loop)
    kernel_str += "}\n"

    tree.body.insert(0, ast.Assign(
                targets=[ast.Name(id='kernel_str', ctx=ast.Load())],
                value=ast.Constant(value=kernel_str) 
                ))
    return tree
