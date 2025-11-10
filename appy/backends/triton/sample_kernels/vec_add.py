import torch
import numpy as np
import triton
import triton.language as tl


@triton.jit
def _kernel0(__range_var0, c, a, b):
    pass
    i = 0 + tl.program_id(0) * 256
    tl.store(
        c + (i + tl.arange(0, 256) + tl.arange(0, 1)),
        tl.load(
            a + (i + tl.arange(0, 256) + tl.arange(0, 1)),
            mask=i + tl.arange(0, 256) < __range_var0,
        )
        + tl.load(
            b + (i + tl.arange(0, 256) + tl.arange(0, 1)),
            mask=i + tl.arange(0, 256) < __range_var0,
        ),
        mask=i + tl.arange(0, 256) < __range_var0,
    )


__cten_a = torch.from_numpy(a)
__cten_b = torch.from_numpy(b)
__cten_c = torch.from_numpy(c)
__gten_a = __cten_a.to("cuda")
__gten_b = __cten_b.to("cuda")
__gten_c = __cten_c.to("cuda")
__range_var0 = a.shape[0]
kernel_grid = lambda META: ((__range_var0 - 0 + 256 - 1) // 256,)
fn = _kernel0[kernel_grid](__range_var0, __gten_c, __gten_a, __gten_b, num_warps=4)
__cten_c.copy_(__gten_c)
