import torch
import triton
import triton.language as tl

def kernel_loop_1(a, a_shape_0, b, c):
    __tc_a = a
    __tg_a = __tc_a.to('cuda')
    __tc_b = b
    __tg_b = __tc_b.to('cuda')
    __tc_c = c
    __tg_c = __tc_c.to('cuda')
    grid = ((a_shape_0 - 0 + (256 - 1)) // 256,)
    _kernel_loop_1[grid](__tg_a, a_shape_0, __tg_b, __tg_c, num_warps=4)
    __tc_c.copy_(__tg_c)

@triton.jit
def _kernel_loop_1(a, a_shape_0, b, c):
    __idx_i = tl.program_id(0) * 256
    i = __idx_i + tl.arange(0, 256)
    tl.store(
        c + i,
        tl.load(a + i, mask=__idx_i + tl.arange(0, 256) < a_shape_0)
        + tl.load(b + i, mask=__idx_i + tl.arange(0, 256) < a_shape_0),
        mask=__idx_i + tl.arange(0, 256) < a_shape_0,
    )
