# Memory Hazards in APPy Metal SIMD Execution Model

## Background

In APPy's Metal backend, a `#pragma parallel for` loop in SIMD mode dispatches
`N × SIMD_WIDTH` (N × 1024) threads for N iterations. Each iteration `i` is
handled cooperatively by a group of 1024 threads (lanes 0–1023):

```
i    = grid_id / SIMD_WIDTH
lane = grid_id % SIMD_WIDTH
```

Within each iteration, reductions (`np.sum/min/max`) are lowered to a strided
accumulation across lanes followed by a threadgroup tree reduction. The final
reduced value is broadcast back to all lanes via threadgroup shared memory:

```c
threadgroup float __threadgroup_s[1024];
// ... strided accumulation into s (per-lane partial sum) ...
__threadgroup_s[lane] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
// ... tree reduction into __threadgroup_s[0] ...
s = __threadgroup_s[0];   // all 1024 lanes now hold the reduced value
```

Any direct write to a scalar device-memory location (e.g. `y[i] = s`) is
guarded so that **only lane 0 performs the write**:

```c
if (lane == 0) { y[i] = s; }
```

---

## Memory Hazard Analysis

### RAW (Read-After-Write) — Critical hazard

**Pattern**: a guarded scalar write to device memory, followed by an all-lane
read of the same location in the same parallel-for iteration.

**Example**:
```python
#pragma parallel for
for j in range(N):
    R[k, j] = np.sum(Q[:, k] * A[:, j])   # guarded write: only lane 0
    A[:, j] -= Q[:, k] * R[k, j]           # all 1024 lanes read R[k, j]
```

After lowering, the write-back `R[k, j] = __reduce_sum_var` is guarded to lane 0.
The subsequent column update reads `R[k, j]` from device memory on all 1024 lanes,
but there is no barrier in between. Lanes 1–1023 will see a stale value.

**Fix 1 — local variable (preferred)**: use the threadgroup-local propagated
reduction variable instead of re-reading from device memory:

```python
#pragma parallel for
for j in range(N):
    s = np.sum(Q[:, k] * A[:, j])   # s broadcast to all 1024 lanes via threadgroup mem
    R[k, j] = s                      # lane 0 writes to device memory (output only)
    A[:, j] -= Q[:, k] * s          # all lanes use local s — no device memory read
```

After the threadgroup tree reduction, `s` is already correct for all lanes via
threadgroup memory (which uses `mem_threadgroup` barrier — fast). No additional
barrier needed.

**Fix 2 — device memory barrier**: insert `threadgroup_barrier(mem_flags::mem_device)`
between the guarded write and the subsequent read, so lane 0's write is flushed
and visible to all other lanes:

```c
if (lane == 0) { R[k * N + j] = __reduce_sum_var; }
threadgroup_barrier(mem_flags::mem_device);   // ensure device memory visibility
... = R[k * N + j];                           // now safe for all lanes
```

Fix 2 allows more natural user code but uses a `mem_device` barrier, which is
significantly more expensive than a `mem_threadgroup` barrier. Fix 1 is
therefore preferred wherever possible.

**Future compiler pass**: detect RAW patterns (guarded write to device memory
followed by a read of the same location) and automatically insert a
`mem_device` barrier, allowing users to write natural code without worrying
about this hazard.

**General rule**: this applies only to **scalar device-memory writes** (single-element
subscript writes guarded by lane 0). Writes inside a simd inner loop
(e.g. `A[i, j] += ...` where `j` is the simd loop variable) are safe because
each lane writes to a distinct element.

---

### WAR (Write-After-Read) — Unlikely but theoretically possible

**Pattern**: all lanes read from a device-memory location, then lane 0 writes
to the same location.

Within a single SIMD group (32 threads on Apple Silicon), program order ensures
the read precedes the write. However, the 1024 threads form 32 independent SIMD
groups. SIMD group 0 (containing lane 0) could advance to the write before
another SIMD group executes its read, causing that group to observe the new
value instead of the old one.

In practice this does not arise in typical npbench kernels because guarded
writes always come after all preceding reads in program order. Worth noting for
more complex kernels.

---

### WAW (Write-After-Write) — No hazard

**Pattern**: two guarded writes to the same scalar memory location.

Since both writes are guarded to lane 0, they execute sequentially in lane 0's
program order. The final value is the second write. No race condition.

---

---

## Array Slice Hazards in SIMD Inner Loops

Array slice assignments inside a SIMD parallel-for body are lowered by
`lower_array_op_to_loop` to a strided inner loop. Two cases arise with
different hazard profiles.

### Case 1 — Slice with index offset (WAR/RAW across loop passes)

**Pattern**: the write index differs from the read index by a constant offset.

**Example**:
```python
#pragma parallel for
for i in range(N):
    A[i, 1:M] = 0.5 * (A[i, :M-1] + B[i, :M-1])
```

Lowered to (conceptually):
```c
// pass p covers elements [p*1024, (p+1)*1024)
for (uint p = 0; p < ceil(M/1024); p++) {
    uint j = p * 1024 + lane;      // write index
    if (j >= 1 && j < M)
        A[i*M + j] = 0.5f * (A[i*M + j - 1] + B[i*M + j - 1]);
}
```

**Hazard — two problems**:

1. *Within a pass*: lane `k` writes `A[i, k]` while lane `k-1` reads `A[i, k-1]`.
   These are distinct indices, so there is no within-pass conflict.

2. *Boundary element across passes*: pass `p` reads elements in
   `[p*1024 - 1, (p+1)*1024 - 1)` and writes to `[p*1024, (p+1)*1024)`.
   The last element written in pass `p` is index `(p+1)*1024 - 1`.
   That same index is **read** by pass `p+1` (as `j - 1` with `j = (p+1)*1024`),
   but by then it already holds the *new* value — a classic read-after-write
   across passes.

   A `threadgroup_barrier(mem_flags::mem_device)` between passes can prevent
   within-pass conflicts (problem 1) but **cannot fix** the boundary element
   problem (problem 2): even if all 1024 threads synchronize before writing, the
   boundary element `(p+1)*1024 - 1` is overwritten during pass `p` and is
   therefore unavailable for pass `p+1` to read its *original* value.

**Real-world example — `seidel_2d`**:
```python
for t in range(TSTEPS - 1):
    for i in range(1, N - 1):
        A[i, 1:-1] += (A[i-1, :-2] + A[i-1, 1:-1] + A[i-1, 2:] +
                       A[i,   2:]   +
                       A[i+1, :-2] + A[i+1, 1:-1] + A[i+1, 2:])
        for j in range(1, N - 1):
            A[i, j] += A[i, j - 1]
            A[i, j] /= 9.0
```
The `A[i, 1:-1] += ...` slice assignment reads `A[i, 2:]` (offset +1) and
writes `A[i, 1:-1]`. The inner `j` loop then reads `A[i, j-1]` (offset -1)
from the already-updated row. Both have cross-iteration dependences that make
in-place parallel execution incorrect. A correct parallel implementation
requires staging the update into a temporary buffer.

**Special case — loop bound ≤ 1024 (single pass, barrier sufficient)**:

When the slice length is at most 1024, all elements fit within a single
threadgroup pass — no strided multi-pass loop is needed. In this case the
boundary element problem disappears entirely: all reads happen before any
writes within the same pass. A single `threadgroup_barrier(mem_flags::mem_device)`
between the read phase and the write phase is sufficient to make it correct:

```c
// Read phase: all 1024 lanes read into registers
float val = 0.5f * (A[i*M + j - 1] + B[i*M + j - 1]);
threadgroup_barrier(mem_flags::mem_device);  // flush reads before writes
// Write phase: all lanes write (no cross-pass boundary issue)
if (j >= 1 && j < M) A[i*M + j] = val;
```

This is worth exploiting in future compiler passes: if the compiler can prove
the slice fits in one threadgroup (length ≤ SIMD_WIDTH), it can emit a
barrier-protected single-pass kernel instead of requiring a double buffer.

**Fix for multi-pass case**: introduce an intermediate (double) buffer — read from
the old array, write to a temporary, then copy back:
```python
#pragma parallel for
for i in range(N):
    tmp[i, 1:M] = 0.5 * (A[i, :M-1] + B[i, :M-1])
# ... then: A[:, 1:M] = tmp[:, 1:M]
```
This is the standard compiler solution for loop-carried dependences with
non-trivial offsets and cannot be avoided within a single in-place parallel pass.

---

### Case 2 — Slice without index offset (safe)

**Pattern**: each lane reads and writes the same index; no cross-lane dependency.

**Example**:
```python
#pragma parallel for
for i in range(N):
    A[i, :] = B[i, :] * 2.0
```

Lowered to:
```c
uint j = p * 1024 + lane;
if (j < M) A[i*M + j] = B[i*M + j] * 2.0f;
```

Lane `k` reads `B[i, k]` and writes `A[i, k]`. Because read and write touch
the same index (no offset), lanes are fully independent and no hazard arises —
even when `A` and `B` alias.

---

## Barrier Comparison: Metal vs CUDA

| Barrier | Execution sync | Threadgroup mem | Device mem |
|---------|---------------|-----------------|------------|
| CUDA `__syncthreads()` | Yes | Yes | Yes |
| Metal `threadgroup_barrier(mem_flags::mem_threadgroup)` | Yes | Yes | No |
| Metal `threadgroup_barrier(mem_flags::mem_device)` | Yes | No | Yes |
| Metal `threadgroup_barrier(mem_flags::mem_threadgroup \| mem_flags::mem_device)` | Yes | Yes | Yes |

CUDA's `__syncthreads()` is equivalent to the last row — it always provides a
full barrier for both shared and global memory within a block. Metal separates
these concerns via `mem_flags`, allowing cheaper barriers when only one memory
space needs to be synced. In our threadgroup reductions, `mem_threadgroup` is
sufficient (and cheaper) because the threadgroup array lives in threadgroup
memory, not device memory.
