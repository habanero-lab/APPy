def is_numpy_array(val):
    return f'{type(val).__module__}.{type(val).__name__}' == 'numpy.ndarray'

def is_torch_cpu_tensor(val):
    return f'{type(val).__module__}.{type(val).__name__}' == 'torch.Tensor' and val.device.type == 'cpu'

def is_torch_tensor(val):
    return f'{type(val).__module__}.{type(val).__name__}' == 'torch.Tensor'

def is_torch_gpu_tensor(val):
    return f'{type(val).__module__}.{type(val).__name__}' == 'torch.Tensor' and val.device.type == 'cuda'