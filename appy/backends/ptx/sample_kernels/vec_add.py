import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initialize CUDA driver context

# Load PTX code from file
with open(f"{os.environ["HOME"]}/projects/APPy/appy/backends/ptx/sample_kernels/vec_add.ptx", "r") as f:
    ptx_code = f.read()

# Load module and get kernel function
module = cuda.module_from_buffer(ptx_code.encode())
kernel = module.get_function("vec_add")


def kernel_appy(a, b, c):
    # Allocate GPU memory
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Launch kernel
    N = a.shape[0]
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    kernel(a_gpu, b_gpu, c_gpu, np.uint32(N),
        block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy result back
    cuda.memcpy_dtoh(c, c_gpu)
