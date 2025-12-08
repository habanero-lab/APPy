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
    for var, val in val_map.items():
        if var == niters:
            continue

        assert hasattr(val, "buf")

        dtype = val.dtype
        if str(dtype) == 'float32':
            dtype = 'float'
        elif str(dtype) == 'int32':
            dtype = 'int'
        else:
            assert False, f"Unsupported array dtype {dtype}"

        s += f"device {dtype}* {var} [[ buffer({count}) ]], "
        count += 1

    s += f"uint id [[ thread_position_in_grid ]])"
    return s

def gen_var_decls(tree, val_map):
    # Check all assigned vars, declare them if they are not in val_map
    var_to_type = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id not in val_map:
                print(ast.unparse(node))
                var_to_type[target.id] = target.appy_type

    s = ""
    for var, ty in var_to_type.items():
        s += f"{ty} {var};\n"
    print(s)
    return s


def gen_func_body(loop_name):
    return f'''
    float xi = x[id];
    float x3 = xi * xi * xi;
    float t = tanh(0.79788456f * (xi + 0.044715f * x3));
    y[id] = 0.5f * xi * (1.0f + t);
'''

def transform(tree, replaced_loop, loop_name, val_map):
    kernel_str = gen_headers()
    kernel_str += gen_func_header(loop_name, replaced_loop, val_map)
    kernel_str += "{\n"
    # gen_var_decls(replaced_loop, val_map)
    kernel_str += gen_func_body(loop_name)
    kernel_str += "}\n"

    tree.body.insert(0, ast.Assign(
                targets=[ast.Name(id='kernel_str', ctx=ast.Load())],
                value=ast.Constant(value=kernel_str) 
                ))
    return tree
