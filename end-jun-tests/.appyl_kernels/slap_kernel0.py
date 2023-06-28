import torch
import triton
import triton.language as tl


@triton.jit
def _kernel0(
    a,
    a_shape_0: tl.constexpr,
    a_stride_0: tl.constexpr,
    b,
    b_shape_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    c,
    c_shape_0: tl.constexpr,
    c_stride_0: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pass
    i = 0 + tl.program_id(0) * BLOCK
    i = i + tl.arange(0, BLOCK)
    tl.store(
        c + i,
        tl.load(a + i, mask=i < a_shape_0, other=0)
        + tl.load(b + i, mask=i < b_shape_0, other=0),
        mask=i < c_shape_0,
    )


def kernel(a, b, c, N, BLOCK=256):
    blockDim_x = (N - 0 + BLOCK - 1) // BLOCK
    fn = _kernel0[blockDim_x,](
        a,
        a.size(0),
        a.stride(0),
        b,
        b.size(0),
        b.stride(0),
        c,
        c.size(0),
        c.stride(0),
        N,
        BLOCK,
    )
