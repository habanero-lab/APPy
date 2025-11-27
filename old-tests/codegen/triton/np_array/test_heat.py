import math
import numpy as np
import pytest

import appy

# ==========================
# Constants and Helpers
# ==========================

PI = math.acos(-1.0)

def initial_value_python(n, dx, length, u):
    for j in range(n):
        y = (j + 1) * dx
        for i in range(n):
            x = (i + 1) * dx
            u[i, j] = math.sin(PI * x / length) * math.sin(PI * y / length)

def zero_python(n, u):
    u[:, :] = 0.0

def solution(t, x, y, alpha, length):
    return math.exp(-2.0 * alpha * PI * PI * t / (length * length)) * \
           math.sin(PI * x / length) * math.sin(PI * y / length)

def l2norm_python(n, u, nsteps, dt, alpha, dx, length):
    time = dt * nsteps
    norm = 0.0
    for j in range(n):
        y = (j + 1) * dx
        for i in range(n):
            x = (i + 1) * dx
            exact = solution(time, x, y, alpha, length)
            diff = u[i, j] - exact
            norm += diff * diff
    return math.sqrt(norm)

# Pure Python reference solver (no APPy, no Numba)
def solve_reference(n, alpha, dx, dt, u, u_tmp, nsteps):
    r = alpha * dt / (dx * dx)
    r2 = 1.0 - 4.0 * r

    for _ in range(nsteps):
        for i in range(n):
            for j in range(n):
                center = u[i, j]
                left   = u[i-1, j] if i > 0 else 0.0
                right  = u[i+1, j] if i < n - 1 else 0.0
                down   = u[i, j-1] if j > 0 else 0.0
                up     = u[i, j+1] if j < n - 1 else 0.0
                u_tmp[i, j] = r2 * center + r * (left + right + down + up)
        u, u_tmp = u_tmp, u
    return u


# ==========================
# APPy Kernel Under Test
# ==========================

@appy.jit(backend='triton', dump_code=True)
def solve_appy(n, alpha, dx, dt, u, u_tmp, nsteps):
    r = alpha * dt / (dx * dx)
    r2 = 1.0 - 4.0 * r
    for _ in range(nsteps):
        #pragma parallel for
        for i in range(n):
            #pragma simd
            for j in range(n):
                center = u[i, j]
                left   = u[i-1, j] if i > 0 else 0.0
                right  = u[i+1, j] if i < n - 1 else 0.0
                down   = u[i, j-1] if j > 0 else 0.0
                up     = u[i, j+1] if j < n - 1 else 0.0
                u_tmp[i, j] = r2 * center + r * (left + right + down + up)
        u, u_tmp = u_tmp, u
    return u


# ==========================
# The Actual Test
# ==========================

def test_heat_solver_small():
    # Small problem for CI
    n = 16
    nsteps = 2
    alpha = 0.1
    length = 100.0
    dx = length / (n + 1)
    dt = 0.01

    # Allocate arrays
    u_ref = np.zeros((n, n), dtype=np.float64)
    u_tmp_ref = np.zeros_like(u_ref)

    u_appy = np.zeros((n, n), dtype=np.float64)
    u_tmp_appy = np.zeros_like(u_appy)

    # Initial conditions
    initial_value_python(n, dx, length, u_ref)
    initial_value_python(n, dx, length, u_appy)

    zero_python(n, u_tmp_ref)
    zero_python(n, u_tmp_appy)

    # Run reference solver
    u_ref = solve_reference(n, alpha, dx, dt, u_ref, u_tmp_ref, nsteps)

    # Run APPy solver
    u_appy = solve_appy(n, alpha, dx, dt, u_appy, u_tmp_appy, nsteps)

    # Compare results
    np.testing.assert_allclose(
        u_appy, u_ref, rtol=1e-6, atol=1e-6,
        err_msg="APPy heat solver did not match reference implementation"
    )

    # Compare L2 norms
    norm_ref = l2norm_python(n, u_ref, nsteps, dt, alpha, dx, length)
    norm_appy = l2norm_python(n, u_appy, nsteps, dt, alpha, dx, length)

    assert abs(norm_ref - norm_appy) < 1e-6
