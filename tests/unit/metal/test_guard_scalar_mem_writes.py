"""
Tests for guard_scalar_mem_writes: scalar writes in the pfor body that are
nested inside if-blocks must be guarded with `if (lane == 0)`. Writes inside
SIMD loops must NOT be guarded (each lane owns a distinct element).

Two execution modes:
- Flat mode (use_simd=False): 1 thread per pfor iteration, no lane variable,
  no guarding needed. Triggered when the pfor body has no simd inner loop.
- SIMD/threadgroup mode (use_simd=True): 1024 threads per pfor iteration,
  scalar writes must be guarded. Triggered by np.sum / array slice ops.
"""
import numpy as np
import appy
import appy.np_shared as nps


# --- Flat mode: plain scalar write, no simd inner loop ---

@appy.jit(dump_code=False)
def kernel_flat(A, out):
    N = A.shape[0]
    #pragma parallel for
    for i in range(N):
        out[i] = A[i] * 2.0


def test_flat_mode_scalar_write():
    N = 64
    rng = np.random.default_rng(0)
    A = nps.copy(rng.random((N,)).astype(np.float32))
    out = nps.zeros((N,), dtype=np.float32)

    kernel_flat(A, out)

    assert np.allclose(out, np.array(A) * 2.0, atol=1e-5)


# --- SIMD mode: scalar write inside an if/else block ---

@appy.jit(dump_code=False)
def kernel_write_in_if(A, out, thresh):
    N = A.shape[0]
    #pragma parallel for
    for i in range(N):
        s = np.sum(A[i, :])
        if s > thresh:
            out[i] = s
        else:
            out[i] = 0.0


def test_simd_scalar_write_inside_if():
    N, M = 64, 32
    rng = np.random.default_rng(1)
    A = nps.copy(rng.random((N, M)).astype(np.float32))
    out = nps.zeros((N,), dtype=np.float32)
    thresh = np.float32(M / 2)

    kernel_write_in_if(A, out, thresh)

    A_np = np.array(A)
    row_sums = A_np.sum(axis=1)
    out_ref = np.where(row_sums > float(thresh), row_sums, 0.0).astype(np.float32)
    assert np.allclose(out, out_ref, atol=1e-4), f"max err {np.max(np.abs(out - out_ref))}"


# --- SIMD mode: scatter write inside a SIMD loop (must NOT be guarded) ---

@appy.jit(dump_code=False)
def kernel_simd_scatter(A, out):
    N, M = A.shape
    #pragma parallel for
    for i in range(N):
        out[i, :] = A[i, :] * 2.0


def test_simd_scatter_not_guarded():
    N, M = 64, 32
    rng = np.random.default_rng(2)
    A = nps.copy(rng.random((N, M)).astype(np.float32))
    out = nps.zeros((N, M), dtype=np.float32)

    kernel_simd_scatter(A, out)

    assert np.allclose(out, np.array(A) * 2.0, atol=1e-5)
