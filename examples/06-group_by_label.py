import torch
import appy
# import cupy
# appy.tensorlib = cupy

torch.set_default_device('cuda')

@appy.jit(auto_simd=True)
def kernel_appy(N, nlabels, nfeatures, data, labels):
    out = torch.zeros([nlabels, nfeatures], dtype=data.dtype)
    count = torch.zeros(nlabels, dtype=data.dtype)
    #pragma parallel for
    for i in range(N):
        l = labels[i]
        #pragma atomic
        out[l, :nfeatures] += data[i, :nfeatures]
        #pragma atomic
        count[l] += 1
    return out / count[:,None]
    

def kernel_lib1(N, nlabels, nfeatures, data, labels):
    out = torch.zeros([nlabels, nfeatures], dtype=data.dtype)
    count = torch.zeros(nlabels, dtype=data.dtype)
    for i in range(N):
        l = labels[i]
        out[l] += data[i]
        count[l] += 1
    return out / count[:,None]


def kernel_lib2(N, nlabels, nfeatures, data, labels):
    out = torch.empty([nlabels, nfeatures], dtype=data.dtype)
    for i in range(nlabels):
        out[i] = data[labels == i].mean(axis=0)
    return out


def test():
    nfeatures = 30
    nlabels = 10
    for N in [1000]: 
        data = torch.randn(N, nfeatures)
        labels = torch.randint(0, nlabels, size=(N,))        
        y_ref = kernel_lib2(N, nlabels, nfeatures, data, labels)
        for f in [kernel_lib1, kernel_lib2, kernel_appy]:
            y = f(N, nlabels, nfeatures, data, labels)
            assert appy.utils.allclose(y, y_ref, atol=1e-6)
            ms = appy.utils.bench(lambda: f(N, nlabels, nfeatures, data, labels))
            print(f"{f.__name__}: {ms:.4f} ms")


if __name__ == "__main__":
    test()
