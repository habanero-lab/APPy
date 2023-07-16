import numpy as np
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
        N = 2
    elif one_run > 5:
        N = 1
    return t0.timeit(N).mean * 1000

def allclose(a, b):
    if isinstance(a, torch.Tensor):
        return torch.allclose(a, b)
    elif isinstance(a, np.array):
        return np.allclose(a, b)
