import numpy as np
import cupy
import torch
import torch.utils.benchmark as torchbench

def bench(fn):
    t0 = torchbench.Timer(
        stmt='fn()',
        globals={'fn': fn},
        num_threads=torch.get_num_threads()
    )
    N = 20
    one_run = t0.timeit(1).mean
    if one_run > 1:
        N = 5
    elif one_run > 5:
        N = 2
    return t0.timeit(N).mean * 1000

def allclose(a, b, verbose=True, rtol=1e-05, atol=1e-06, equal_nan=False):
    assert type(a) == type(b)
    if isinstance(a, np.ndarray):
        f = np.allclose
        max = np.max
    if isinstance(a, cupy.ndarray):
        f = cupy.allclose
        max = cupy.max
    elif isinstance(b, torch.Tensor):
        f = torch.allclose
        max = torch.max
    else:
        assert False

    if not f(a, b, rtol, atol) and verbose:
        diff = a - b
        # print(torch.where(diff != 0))
        # print(diff[diff != 0])
        if len(a.shape) < 2:
            print(a)
            print(b)
        else:
            print(a[0])
            print(b[0])
        print('max diff:', max(diff))

    return f(a, b, rtol, atol)
