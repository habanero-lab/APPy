# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(TMAX, NX, NY, datatype=np.float64):
    ex = np.fromfunction(lambda i, j: (i * (j + 1)) / NX, (NX, NY),
                         dtype=datatype)
    ey = np.fromfunction(lambda i, j: (i * (j + 2)) / NY, (NX, NY),
                         dtype=datatype)
    hz = np.fromfunction(lambda i, j: (i * (j + 3)) / NX, (NX, NY),
                         dtype=datatype)
    _fict_ = np.fromfunction(lambda i: i, (TMAX, ), dtype=datatype)

    return ex, ey, hz, _fict_


def kernel_np(TMAX, ex, ey, hz, _fict_):

    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] -
                               ey[:-1, :-1])
    return ey, ex, hz


import appy

@appy.jit(backend="triton", dump_code=True)
def kernel_appy(TMAX, ex, ey, hz, _fict_):

    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        #pragma parallel for
        for i in range(1, ey.shape[0]):
            #pragma simd
            for j in range(ey.shape[1]):
                ey[i, j] -= 0.5 * (hz[i, j] - hz[i - 1, j])
        #pragma parallel for
        for i in range(ex.shape[0]):
            #pragma simd
            for j in range(1, ex.shape[1]):
                ex[i, j] -= 0.5 * (hz[i, j] - hz[i, j - 1])
        #pragma parallel for
        for i in range(hz.shape[0]-1):
            #pragma simd
            for j in range(hz.shape[1]-1):
                hz[i, j] -= 0.7 * (ex[i, j+1] - ex[i, j] + ey[i+1, j] -
                                   ey[i, j])
    return ey, ex, hz


def test():
    TMAX = 30
    NX = 100
    NY = 100
    ex, ey, hz, _fict_ = initialize(TMAX, NX, NY)
    
    ex1, ey1, hz1 = ex.copy(), ey.copy(), hz.copy()
    ex2, ey2, hz2 = ex.copy(), ey.copy(), hz.copy()

    kernel_np(TMAX, ex1, ey1, hz1, _fict_)
    kernel_appy(TMAX, ex2, ey2, hz2, _fict_)

    assert np.allclose(ex1, ex2)
    assert np.allclose(ey1, ey2)
    assert np.allclose(hz1, hz2)