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
