"""
Benchmark: APPy CUDA backend vs APPy Triton backend.

Both backends receive torch GPU tensors so no host<->device transfer is
included in the timings.  Results are median kernel wall-time over N_RUNS.
"""

import torch
import numpy as np
import appy
from time import perf_counter

N_RUNS  = 50
WARMUP  = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bench(fn, *args, label=""):
    # warmup
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append(perf_counter() - t0)

    times.sort()
    median_ms = times[len(times) // 2] * 1e3
    return label, median_ms


def print_table(title, size_str, rows):
    col_w = 24
    print(f"\n┌─ {title}  [{size_str}] {'─' * (col_w - len(title) - len(size_str) - 4)}┐")
    print(f"│  {'Variant':<{col_w}}  {'Median (ms)':>12} │")
    print(f"│  {'─' * col_w}  {'─' * 12} │")
    for label, ms in rows:
        print(f"│  {label:<{col_w}}  {ms:>11.4f}  │")
    print(f"└{'─' * (col_w + 18)}┘")


def torch_tensor(*shape, dtype=torch.float32):
    return torch.rand(*shape, dtype=dtype, device='cuda')

# ---------------------------------------------------------------------------
# 1. SAXPY   y = a*x + y
# ---------------------------------------------------------------------------

@appy.jit(backend="cuda", dump_code=False)
def saxpy_cuda(a, x, y):
    #pragma parallel for
    for i in range(x.shape[0]):
        y[i] = a * x[i] + y[i]


@appy.jit(backend="triton", dump_code=False)
def saxpy_triton(a, x, y):
    #pragma parallel for simd
    for i in range(x.shape[0]):
        y[i] = a * x[i] + y[i]


def bench_saxpy(N):
    a  = 2.5
    x  = torch_tensor(N)
    yc = torch_tensor(N)
    yt = torch_tensor(N)

    rows = [
        bench(saxpy_cuda,                    a, x, yc, label="CUDA"),
        bench(saxpy_triton,                  a, x, yt, label="Triton"),
        bench(lambda: torch.add(x, x, out=yc),         label="torch baseline"),
    ]
    print_table("saxpy  y=a*x+y", f"N={N:,}", rows)


# ---------------------------------------------------------------------------
# 2. Matrix-vector  y[i] = sum_j A[i,j]*x[j]   (row-parallel)
# ---------------------------------------------------------------------------

@appy.jit(backend="cuda", dump_code=False)
def matvec_cuda(A, x, y):
    #pragma parallel for
    for i in range(A.shape[0]):
        s = 0.0
        for j in range(A.shape[1]):
            s += A[i, j] * x[j]
        y[i] = s


@appy.jit(backend="triton", dump_code=False)
def matvec_triton(A, x, y):
    #pragma parallel for
    for i in range(A.shape[0]):
        s = 0.0
        #pragma simd
        for j in range(A.shape[1]):
            s += A[i, j] * x[j]
        y[i] = s


@appy.jit(backend="cuda", dump_code=False)
def matvec_cuda_v2(A, x, y):
    #pragma parallel for
    for i in range(A.shape[0]):
        y[i] = np.sum(A[i, :] * x[:])


def bench_matvec(M, N):
    A  = torch_tensor(M, N)
    x  = torch_tensor(N)
    yc = torch_tensor(M)
    yt = torch_tensor(M)

    rows = [
        bench(matvec_cuda,                   A, x, yc, label="CUDA scalar reduce"),
        bench(matvec_cuda_v2,                A, x, yc, label="CUDA array expr"),
        bench(matvec_triton,                 A, x, yt, label="Triton"),
        bench(lambda: torch.mv(A, x, out=yc),          label="torch baseline"),
    ]
    print_table("matvec  y=A@x", f"M={M:,} N={N:,}", rows)


# ---------------------------------------------------------------------------
# 3. Jacobi-2D stencil  (simd inner loop)
# ---------------------------------------------------------------------------

@appy.jit(backend="cuda", dump_code=False)
def jacobi2d_cuda(A, B):
    #pragma parallel for
    for i in range(1, A.shape[0] - 1):
        #pragma simd
        for j in range(1, A.shape[1] - 1):
            B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] +
                              A[i-1, j] + A[i+1, j])


@appy.jit(backend="triton", dump_code=False)
def jacobi2d_triton(A, B):
    #pragma parallel for
    for i in range(1, A.shape[0] - 1):
        #pragma simd
        for j in range(1, A.shape[1] - 1):
            B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] +
                              A[i-1, j] + A[i+1, j])


@appy.jit(backend="cuda", dump_code=False)
def jacobi2d_cuda_v2(A, B):
    #pragma parallel for
    for i in range(1, A.shape[0] - 1):
        B[i, 1:-1] = 0.2 * (A[i, 1:-1] + A[i, :-2] + A[i, 2:] +
                             A[i-1, 1:-1] + A[i+1, 1:-1])


def bench_jacobi2d(N):
    A  = torch_tensor(N, N)
    Bc = torch_tensor(N, N)
    Bt = torch_tensor(N, N)

    rows = [
        bench(jacobi2d_cuda,    A, Bc, label="CUDA simd pragma"),
        bench(jacobi2d_cuda_v2, A, Bc, label="CUDA array expr"),
        bench(jacobi2d_triton,  A, Bt, label="Triton"),
    ]
    print_table("jacobi-2D stencil", f"N={N:,}×{N:,}", rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("APPy  CUDA (pycuda)  vs  Triton  —  torch GPU tensor inputs")
    print(f"warmup={WARMUP} runs={N_RUNS}  metric=median wall-time (ms)")
    print("=" * 60)

    bench_saxpy(N=10_000_000)
    bench_matvec(M=4096, N=4096)
    bench_jacobi2d(N=2048)

    print()
