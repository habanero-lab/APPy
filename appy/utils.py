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

def allclose(a, b, verbose=True, rtol=1e-05, atol=1e-06, equal_nan=False):
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    a, b = a.to('cuda'), b.to('cuda')
    
    if not torch.allclose(a, b, rtol, atol) and verbose:
        #print(a)
        #print(b)
        diff = a - b
        #print(torch.nonzero(diff != 0))
        print(torch.sort(diff[diff != 0]))
    return torch.allclose(a, b, rtol, atol)
