import torch
from dlpack import asdlpack

def get_random_1d_tensors(
    num_tuples,
    tuple_size=1,
    lib=None,
    shape_low_bound=1024,
    shape_up_bound=128 * 1024 * 1024,
    dtypes=[torch.float32, torch.float64, torch.int32],
    device="cuda",
):
    shapes = sorted(torch.randint(shape_low_bound, shape_up_bound, size=(num_tuples,)))
    # print(shapes)
    tensors = []
    for dtype in dtypes:
        for N in shapes:
            tup = []
            for _ in range(tuple_size):
                if dtype in [torch.int32]:
                    tup.append(
                        torch.randint(0, N, size=(N,), device=device, dtype=dtype)
                    )
                else:
                    tup.append(torch.randn(N, device=device, dtype=dtype))
                    
            for i in range(len(tup)):
                if lib != None:
                    tup[i] = lib.from_dlpack(asdlpack(tup[i]))

            if len(tup) == 1:
                tensors.append(tup[0])
            else:
                tensors.append(tup)
    
    return tensors


def get_random_2d_tensors(
    num_tuples,
    tuple_size=1,
    lib="torch",
    M_bound=(1024, 16 * 1024),
    N_bound=(1024, 16 * 1024),
    dtypes=[torch.float32, torch.float64],
    device="cuda",
):
    assert lib == "torch"
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
    # print(tensors)
    return tensors
