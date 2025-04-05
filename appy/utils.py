import torch
import torch.utils.benchmark as torchbench

def bench(fn):
    t0 = torchbench.Timer(
        stmt='fn()',
        globals={'fn': fn},
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
    if type(a) == float:
        import numpy as np
        f = np.allclose
        max = np.max
        a, b = np.array([a]), np.array([b])
    
    if f"{type(a).__module__}.{type(a).__name__}" == 'numpy.ndarray':
        import numpy as np
        f = np.allclose
        max = np.max
    elif f"{type(a).__module__}.{type(a).__name__}" == 'cupy.ndarray':
        import cupy
        f = cupy.allclose
        max = cupy.max
    elif f"{type(a).__module__}.{type(a).__name__}" == 'torch.Tensor':
        f = torch.allclose
        max = torch.max
    else:
        assert False, f'Unsupported type, a: {type(a)}, b: {type(b)}'

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
