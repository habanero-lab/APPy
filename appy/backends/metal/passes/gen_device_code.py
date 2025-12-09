import ast

def gen_headers():
    return '''
#include <metal_stdlib>
using namespace metal;

'''

def gen_func_header(loop_name, replaced_loop, val_map):
    s = f"kernel void _{loop_name}("

    niters = ast.unparse(replaced_loop.iter.args[0])
    count = 0
    py_to_cpp = {
        'int': 'int',
        'float': 'float',
        'float32': 'float',
        'int32': 'int',
    }
    for var, val in val_map.items():
        if var == niters:
            continue

        if hasattr(val, "buf"):
            # Arrays
            dtype = val.dtype
            s += f"device {py_to_cpp[str(dtype)]}* {var} [[ buffer({count}) ]], "
        else:
            # Scalars
            ty = type(val).__name__
            assert ty in py_to_cpp, f"Unsupported type {ty} for variable {var}"
            s += f"constant {py_to_cpp[ty]}& {var} [[ buffer({count}) ]], "
        count += 1

    s += f"uint grid_id [[ thread_position_in_grid ]])"
    return s

def gen_var_decls(loop, val_map):
    # Check all assigned vars, declare them if they are not in val_map
    var_to_type = {}
    for node in ast.walk(loop):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id not in val_map:
                print(ast.unparse(node))
                var_to_type[target.id] = target.appy_type

    s = ""
    for var, ty in var_to_type.items():
        s += f"    {ty} {var};\n"
    
    loop_index = loop.target.id
    s += f"    uint {loop_index} = grid_id;\n"
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

def transform(tree, replaced_loop, loop_name, val_map):
    from .device_passes import rewrite_func_calls
    from .device_passes import rewrite_multi_dim_indexing
    replaced_loop = rewrite_func_calls.transform(replaced_loop)
    replaced_loop = rewrite_multi_dim_indexing.transform(replaced_loop, val_map)


    kernel_str = gen_headers()
    kernel_str += gen_func_header(loop_name, replaced_loop, val_map)
    kernel_str += "{\n"
    kernel_str += gen_var_decls(replaced_loop, val_map)
    kernel_str += gen_func_body(replaced_loop)
    kernel_str += "}\n"

    tree.body.insert(0, ast.Assign(
                targets=[ast.Name(id='kernel_str', ctx=ast.Load())],
                value=ast.Constant(value=kernel_str) 
                ))
    return tree
