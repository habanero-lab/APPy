import torch

def get_1d_tensors_assorted_shapes(
        num_tuples,
        tuple_size=1,
        lib='torch', 
        shape_low_bound=1024,
        shape_up_bound=128*1024*1024,
        dtypes=[torch.float32, torch.float64],
        device='cuda'
    ):
    assert lib == 'torch'
    shapes = sorted(torch.randint(shape_low_bound, shape_up_bound, size=(num_tuples,)))
    #print(shapes) 
    tensors = []
    for dtype in dtypes:
        for N in shapes:
            tup = []
            for _ in range(tuple_size):
                tup.append(torch.randn(N, device=device, dtype=dtype))
            tensors.append(tup)
    #print(tensors)
    return tensors

def get_2d_tensors_assorted_shapes(
        num_tuples,
        tuple_size=1,
        lib='torch', 
        M_bound=(1024, 16*1024),
        N_bound=(1024, 16*1024),
        dtypes=[torch.float32, torch.float64],
        device='cuda'
    ):
    assert lib == 'torch'
    shapes = []
    for _ in range(num_tuples):
        M = torch.randint(M_bound[0], M_bound[1], size=(1,))
        N = torch.randint(N_bound[0], N_bound[1], size=(1,))
        shapes.append((M, N))

    print(shapes) 
    tensors = []
    for dtype in dtypes:
        for M, N in shapes:
            tup = []
            for _ in range(tuple_size):
                tup.append(torch.randn(M, N, device=device, dtype=dtype))
            tensors.append(tup)
    #print(tensors)
    return tensors
