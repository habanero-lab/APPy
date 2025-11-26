# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, N), dtype=np.float64)
    return x


def go_fast_np(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


import appy
import math
@appy.jit(backend="triton", dump_code=True)
def go_fast_appy(a):
    trace = 0.0
    #pragma parallel for simd shared(trace)
    for i in range(a.shape[0]):
        trace += math.tanh(a[i, i])
    print("trace", trace)
    return a + trace


def test():
    N = 100

    x = initialize(N)
    x_ref = go_fast_np(x)
    x = go_fast_appy(x)
    assert np.allclose(x, x_ref, atol=1e-6)