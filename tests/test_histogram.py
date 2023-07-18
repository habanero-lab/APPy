import numpy as np
import torch
from appy import parallel, shared, prange, syncthreads
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule

from torch import arange, zeros, empty

def kernel(a, b, N, nbins: parallel, BLOCK: parallel):
    for i in range(0, N, BLOCK):  #pragma block parallel reduction(+:nbins)
        buf: shared = zeros([nbins], device=a.device, dtype=a.dtype)
        for j in range(0, BLOCK):  #pragma thread parallel reduction(+:buf)
            id = a[i+j]
            buf[id] += 1
        b[:nbins] += buf[:nbins]
        
