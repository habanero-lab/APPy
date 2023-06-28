import torch
import triton
import triton.language as tl


@triton.jit
def _kernel0(
    a,
    a_shape_0: tl.constexpr,
    a_stride_0: tl.constexpr,
    a_shape_1: tl.constexpr,
    a_stride_1: tl.constexpr,
    b,
    b_shape_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    c,
    c_shape_0: tl.constexpr,
    c_stride_0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pass
    i = 0 + tl.program_id(0) * BM
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for j in range(0, N, BN):
        acc = (
            acc
            + tl.load(
                a
                + ((i + tl.arange(0, BM)) * a_stride_0)[:, None]
                + (j + tl.arange(0, BN))[None, :]
            )
            * tl.load(
                b + j + tl.arange(0, BN), mask=j + tl.arange(0, BN) < b_shape_0, other=0
            )[None, :]
        )
    tl.store(
        c + i + tl.arange(0, BM),
        tl.sum(acc, axis=1),
        mask=i + tl.arange(0, BM) < c_shape_0,
    )


def kernel(a, b, c, M, N, BM=8, BN=256):
    blockDim_x = (M - 0 + BM - 1) // BM
    fn = _kernel0[blockDim_x,](
        a,
        a.size(0),
        a.stride(0),
        a.size(1),
        a.stride(1),
        b,
        b.size(0),
        b.stride(0),
        c,
        c.size(0),
        c.stride(0),
        M,
        N,
        BM,
        BN,
    )
