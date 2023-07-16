import numpy as np
import torch
import torch.utils.benchmark as torchbench

def bench(fn):
    t0 = torchbench.Timer(
        stmt='fn()',
        globals={'fn': fn},
        num_threads=torch.get_num_threads()
    )
    return t0.timeit(20).mean * 1000

def allclose(a, b):
    if isinstance(a, torch.Tensor):
        return torch.allclose(a, b)
    elif isinstance(a, np.array):
        return np.allclose(a, b)
