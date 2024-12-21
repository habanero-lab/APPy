import torch
import appy

torch.set_default_device('cuda')

@appy.jit  
def kernel_appy1(a, mask):
    ## Zero-initialize the output array
    b = torch.zeros(1, dtype=a.dtype)
    N = a.shape[0]
    #pragma parallel for
    for i in range(N):
        if mask[i]:  ## simd directive is not support yet for control flows
            #pragma atomic
            b[0] += a[i]
    return b


@appy.jit  
def kernel_appy2(a, mask):
    ## Zero-initialize the output array
    b = torch.zeros(1, dtype=a.dtype)
    N = a.shape[0]
    #pragma parallel for simd
    for i in range(N):
        #pragma atomic
        b[0] += torch.where(mask[i], a[i], 0.0)
    return b


def kernel_lib(a, mask):
    return a[mask].sum()


def test():
    for N in [10000, 100000, 1000000, 10000000]:
        a = torch.randn(N)
        mask = torch.randint(0, 2, size=(N,)) > 0
        c_ref = kernel_lib(a, mask)
        print(f"N: {a.shape[0]}, dtype: {a.dtype}")
        for f in [kernel_lib, kernel_appy1, kernel_appy2]:
            c = f(a, mask)
            assert appy.utils.allclose(c, c_ref, atol=1e-6)
            ms = appy.utils.bench(lambda: f(a, mask))
            print(f"{f.__name__}: {ms:.4f} ms")

if __name__ == '__main__':
    test()