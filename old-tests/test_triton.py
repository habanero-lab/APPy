import numpy as np

a = np.ones(8)
b = np.ones(8)
c = np.empty_like(a)
a_shape_0 = a.shape[0]

import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a, a_shape_0, b, c):
    pass
    i = 0 + tl.program_id(0) * 256
    tl.store(c + (i + tl.arange(0, 256) + tl.arange(0, 1)), tl.load(a + (i + tl.arange(0, 256) + tl.arange(0, 1)), mask=i + tl.arange(0, 256) < a_shape_0) + tl.load(b + (i + tl.arange(0, 256) + tl.arange(0, 1)), mask=i + tl.arange(0, 256) < a_shape_0), mask=i + tl.arange(0, 256) < a_shape_0)

__tc_a = torch.from_numpy(a)
__tg_a = __tc_a.to('cuda')
__tc_b = torch.from_numpy(b)
__tg_b = __tc_b.to('cuda')
__tc_c = torch.from_numpy(c)
__tg_c = __tc_c.to('cuda')
#pragma parallel for simd
grid = ((a_shape_0 - 0 + (256 - 1)) // 256,)
kernel[grid](__tg_a, a_shape_0, __tg_b, __tg_c, num_warps=4)
__tc_c.copy_(__tg_c)
print(__tc_c)
