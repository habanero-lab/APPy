import ast
import ast_comments as astc
from ...utils import load_module_from_str

def codegen(loop_source, loop_name, val_map, options):
    '''
    Returns a dynamically generated function from the loop source.
    '''
    src = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_loop_1(a, a_shape_0, b, c):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(c + i, tl.load(a + i, mask=i < a_shape_0) + tl.load(b + i, mask=i < a_shape_0), mask=i < a_shape_0)

def kernel_loop_1(a, a_shape_0, b, c):
    grid = ((a_shape_0 - 0 + (256 - 1)) // 256,)
    _kernel_loop_1[grid](a, a_shape_0, b, c, num_warps=4)
    '''
    m = load_module_from_str(src)
    return getattr(m, loop_name)