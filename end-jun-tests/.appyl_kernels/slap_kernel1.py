import torch
import triton
import triton.language as tl


@triton.jit
def _kernel0(
    x,
    x_shape_0: tl.constexpr,
    x_stride_0: tl.constexpr,
    x_shape_1: tl.constexpr,
    x_stride_1: tl.constexpr,
    labels,
    labels_shape_0: tl.constexpr,
    labels_stride_0: tl.constexpr,
    centers,
    centers_shape_0: tl.constexpr,
    centers_stride_0: tl.constexpr,
    centers_shape_1: tl.constexpr,
    centers_stride_1: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    Bj: tl.constexpr,
):
    pass
    i = 0 + tl.program_id(0) * 1
    j = 0 + tl.program_id(1) * 128
    j = j + tl.arange(0, 128)
    label = tl.load(labels + i, mask=i < labels_shape_0, other=0)
    tl.atomic_add(
        centers + label * centers_stride_0 + j, tl.load(x + i * x_stride_0 + j)
    )


def kernel(x, labels, centers, M, N, Bj):
    blockDim_x = M
    blockDim_y = (N - 0 + 128 - 1) // 128
    fn = _kernel0[blockDim_x, blockDim_y](
        x,
        x.size(0),
        x.stride(0),
        x.size(1),
        x.stride(1),
        labels,
        labels.size(0),
        labels.stride(0),
        centers,
        centers.size(0),
        centers.stride(0),
        centers.size(1),
        centers.stride(1),
        M,
        N,
        Bj,
    )
