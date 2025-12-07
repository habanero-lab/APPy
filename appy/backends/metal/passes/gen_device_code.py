import ast

def gen_headers():
    return '''
#include <metal_stdlib>
using namespace metal;

'''

def gen_func_header(loop_name):
    return f'''
kernel void _{loop_name}(
    const device float* x [[ buffer(0) ]],
    device float* y [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]])

'''

def gen_func_body(loop_name):
    return f'''
    float xi = x[id];
    float x3 = xi * xi * xi;
    float t = tanh(0.79788456f * (xi + 0.044715f * x3));
    y[id] = 0.5f * xi * (1.0f + t);
'''

def transform(tree, replaced_loop, loop_name, val_map):
    kernel_str = gen_headers()
    kernel_str += gen_func_header(loop_name)
    kernel_str += "{\n"
    kernel_str += gen_func_body(loop_name)
    kernel_str += "}\n"

    tree.body.insert(0, ast.Assign(
                targets=[ast.Name(id='kernel_str', ctx=ast.Load())],
                value=ast.Constant(value=kernel_str) 
                ))
    return tree
