import torch
import triton
import triton.language as tl


@triton.jit
def _kernel0(
    a_rowptrs,
    a_rowptrs_shape_0: tl.constexpr,
    a_rowptrs_stride_0: tl.constexpr,
    a_cols,
    a_cols_shape_0: tl.constexpr,
    a_cols_stride_0: tl.constexpr,
    a_vals,
    a_vals_shape_0: tl.constexpr,
    a_vals_stride_0: tl.constexpr,
    b,
    b_shape_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    b_shape_1: tl.constexpr,
    b_stride_1: tl.constexpr,
    c,
    c_shape_0: tl.constexpr,
    c_stride_0: tl.constexpr,
    c_shape_1: tl.constexpr,
    c_stride_1: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
    acc_dtype: tl.constexpr,
):
    pass
    i = 0 + tl.program_id(0) * 1
    j = 0 + tl.program_id(1) * BN
    acc = tl.zeros([BN], dtype=acc_dtype)
    for ki in range(
        tl.load(a_rowptrs + i, mask=i < a_rowptrs_shape_0, other=0),
        tl.load(a_rowptrs + i + 1, mask=i + 1 < a_rowptrs_shape_0, other=0),
    ):
        a_ik = tl.load(a_vals + ki, mask=ki < a_vals_shape_0, other=0)
        ks = tl.load(a_cols + ki, mask=ki < a_cols_shape_0, other=0)
        acc = acc + a_ik * tl.load(b + ks * b_stride_0 + j + tl.arange(0, BN))
    tl.store(c + i * c_stride_0 + j + tl.arange(0, BN), acc)


def kernel(a_rowptrs, a_cols, a_vals, b, c, M, K, N, BN, acc_dtype):
    blockDim_x = M
    blockDim_y = (N - 0 + BN - 1) // BN
    fn = _kernel0[blockDim_x, blockDim_y](
        a_rowptrs,
        a_rowptrs.size(0),
        a_rowptrs.stride(0),
        a_cols,
        a_cols.size(0),
        a_cols.stride(0),
        a_vals,
        a_vals.size(0),
        a_vals.stride(0),
        b,
        b.size(0),
        b.stride(0),
        b.size(1),
        b.stride(1),
        c,
        c.size(0),
        c.stride(0),
        c.size(1),
        c.stride(1),
        M,
        K,
        N,
        BN,
        tl.float32,
    )
