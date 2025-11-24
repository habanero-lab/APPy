# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    data, radius = rng.random((N, )), rng.random((N, ))
    return data, radius


import numpy as np


def kernel_np(data, radius, npt):
    rmax = radius.max()
    res = np.zeros(npt, dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))        
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
        # mask_r12 = mask_r12.astype(data.dtype)
        # data_sum = (data * mask_r12).sum()
        # mask_sum = mask_r12.sum()
        # res[i] = data_sum / mask_sum
    return res


import appy

@appy.jit(backend="triton", dump_code=True)
def kernel_appy(data, radius, npt):
    rmax = radius.max()
    res = np.zeros(npt, dtype=np.float64)
    #pragma parallel for
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        s, count = 0.0, 0
        #pragma simd
        for j in range(radius.shape[0]):
            mask_r12 = (r1 <= radius[j]) & (radius[j] < r2)
            s += data[j] if mask_r12 else 0.0
            count += 1 if mask_r12 else 0
        res[i] = s / count
    return res


def test():
    N = 2000
    data, radius = initialize(N)
    npt = 100
    
    res1 = kernel_np(data, radius, npt)
    res2 = kernel_appy(data, radius, npt)

    mask = ~np.isclose(res1, res2, atol=1e-6)
    diff_indices = np.where(mask)[0]
    print(diff_indices)
    print(res1[0], res2[0])
    assert len(diff_indices) == 0




'''
benchmark info:

{
    "benchmark": {
        "name": "Azimuthal Integration - Naive",
        "short_name": "azimnaiv",
        "relative_path": "azimint_naive",
        "module_name": "azimint_naive",
        "func_name": "azimint_naive",
        "kind": "microapp",
        "domain": "Physics",
        "dwarf": "spectral_methods",
        "parameters": {
            "S": { "N": 400000, "npt": 1000 },
            "M": { "N": 4000000, "npt": 1000 },
            "L": { "N": 40000000, "npt": 1000 },

            "paper": { "N": 1280000, "npt": 200 }
        },
        "init": {
            "func_name": "initialize",
            "input_args": ["N"],
            "output_args": ["data", "radius"]
        },
        "input_args": ["data", "radius", "npt"],
        "array_args": ["data", "radius"],
        "output_args": []
    }
}
'''