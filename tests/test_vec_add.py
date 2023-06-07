import torch
import bmp
from bmp import parallel
import triton
from kernel_tuner import tune_kernel

kernel_string = """
__global__ void vector_add(float *c, float *a, float *b, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""


#@bmp.jit
def kernel(a, b, c, BLOCK: parallel):
    for i in range(0, a.shape[0], BLOCK):  #pragma parallel
        idx = range(i, i+BLOCK)
        c[idx] = a[idx] + b[idx]
        

for shape in [1024*128, 1024*1024]:
    N = shape
    print(f'N: {N}')
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    ms, _, _ = triton.testing.do_bench(lambda: a + b)
    print(f'torch: {ms} ms')


    for f in [kernel]:
        c = torch.zeros_like(a)
        
        tune_params = dict()
        tune_params["block_size_x"] = [32, 64, 128, 256, 512]
        kernel = tune_kernel("vector_add", kernel_string, N, (c, a, b, torch.tensor(N)), tune_params)
        print(type(kernel))
        
        BLOCK = 128
        f(a, b, c, BLOCK)
        assert(torch.allclose(c, a+b))
        ms, _, _ = triton.testing.do_bench(lambda: f(a, b, c, BLOCK))
        print(f'kernel: {ms} ms')

