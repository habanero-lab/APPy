import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum

def slap_kernel(a, N, BLOCK):
    b = torch.zeros(1, device=a.device, dtype=a.dtype)
    _slap_kernel(a, b, N, BLOCK)
    return b

@slap.jit
def _slap_kernel(a, b, N, BLOCK):
    #pragma parallel reduction(b)
    for i in range(0, N, BLOCK):  
        b[0] += sum(a[i:i+BLOCK])

def slap_kernel1(a, N, BLOCK):
    b = torch.zeros(1, device=a.device, dtype=a.dtype)
    _slap_kernel1(a, b, N, BLOCK)
    return b

@slap.jit
def _slap_kernel1(a, b, N, BLOCK):
    #pragma parallel reduction(b) block(512)
    for i in range(N):  
        b[0] += a[i]

@triton.jit
def _triton_kernel(a, b, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK
    _t0 = i + tl.arange(0, BLOCK)
    _t1 = tl.load(a+_t0)
    _t2 = tl.sum(_t1, axis=0)
    tl.atomic_add(b+0, _t2)

def triton_kernel(a, N, BLOCK):
    b = torch.zeros(1, device=a.device, dtype=a.dtype)
    # Start parallel for loop
    nblocks = (N+BLOCK-1) // BLOCK
    _triton_kernel[(nblocks,)](a, b, N, BLOCK)
    return b

def torch_kernel(a, N, BLOCK):
    b = sum(a)
    return b
    
def test1():
    for dtype in [torch.float32, torch.float64]:
        for shape in [1024*128, 1024*1024, 10*1024*1024]:
            N = shape
            print(f'N: {N}')
            a = torch.randn(N, device='cuda', dtype=dtype)
            b_ref = torch_kernel(a, N, None)

            for f in (torch_kernel, slap_kernel, slap_kernel1):
                BLOCK = 128 * 4
                b = f(a, N, BLOCK)
                assert(torch.allclose(b, b_ref, atol=1e-2))
                ms, _, _ = triton.testing.do_bench(lambda: f(a, N, BLOCK))
                print(f'{f.__name__}: {ms:.4f} ms')

if __name__ == '__main__':
    test1()