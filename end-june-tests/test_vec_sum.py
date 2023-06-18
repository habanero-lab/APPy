import torch
import triton
import triton.language as tl
import slap
from torch import arange, zeros, empty, sum

@slap.jit
def slap_kernel0(a):
    N = a.shape[0]
    b = torch.zeros(1, device=a.device, dtype=a.dtype)
    BLOCK = 512
    #pragma parallel reduction(b)
    for i in range(0, N, BLOCK):  
        b[0] += sum(a[i:i+BLOCK])
    return b

@slap.jit
def slap_kernel(a):
    N = a.shape[0]
    b = torch.zeros(1, device=a.device, dtype=a.dtype)
    #pragma parallel reduction(b) block(512)
    for i in range(N):  
        b[0] += a[i]
    return b


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

def torch_kernel(a):
    b = sum(a)
    return b
    
def test1():
    for shape in [1024*128, 1024*1024, 10*1024*1024]:
        N = shape
        print(f'N: {N}')
        a = torch.randn(N, device='cuda', dtype=torch.float32)
        b_ref = torch_kernel(a)

        for f in (torch_kernel, slap_kernel, slap_kernel0):
            b = f(a)
            assert(torch.allclose(b, b_ref, atol=1e-2))
            ms, _, _ = triton.testing.do_bench(lambda: f(a))
            print(f'{f.__name__}: {ms:.4f} ms')
