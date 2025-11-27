import torch
import triton
import triton.language as tl


def kernel_loop_1(a, a_shape_0, a_shape_1, b, c):
    __tc_a = torch.from_numpy(a)
    __tg_a = __tc_a.to("cuda")
    __tc_b = torch.from_numpy(b)
    __tg_b = __tc_b.to("cuda")
    __tc_c = torch.from_numpy(c)
    __tg_c = __tc_c.to("cuda")
    # pragma parallel for
    grid = ((a_shape_0 - 0 + (1 - 1)) // 1,)
    _kernel_loop_1[grid](
        __tg_a,
        a_shape_0,
        a_shape_1,
        __tg_b,
        __tg_c,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        num_warps=4,
    )
    __tc_c.copy_(__tg_c)


@triton.jit
def _kernel_loop_1(a, a_shape_0, a_shape_1, b, c, a_stride_0, b_stride_0, c_stride_0):
    i = tl.program_id(0) * 1
    for j in range(0, a_shape_1, 1):
        tl.store(
            c + (i * c_stride_0 + j),
            tl.load(a + (i * a_stride_0 + j), mask=None)
            + tl.load(b + (i * b_stride_0 + j), mask=None),
            mask=None,
        )
