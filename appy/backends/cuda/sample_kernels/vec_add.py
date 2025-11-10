import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

# Upload to GPU automatically
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.empty_like(a_gpu)

# ------------------------------
# 2. Define the CUDA kernel (double precision)
# ------------------------------
kernel_code = """
__global__ void vec_add(const double *a, const double *b, double *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}
"""

mod = SourceModule(kernel_code, options=["--use_fast_math"])
vec_add = mod.get_function("vec_add")

# ------------------------------
# 3. Launch configuration
# ------------------------------
N = a.shape[0]
block_size = 256
grid_size = (N + block_size - 1) // block_size

# ------------------------------
# 4. Launch the kernel
# ------------------------------
vec_add(
    a_gpu, b_gpu, c_gpu, np.int32(N),
    block=(block_size, 1, 1),
    grid=(grid_size, 1, 1)
)

# ------------------------------
# 5. Download result and verify
# ------------------------------
c_gpu.get(c)
