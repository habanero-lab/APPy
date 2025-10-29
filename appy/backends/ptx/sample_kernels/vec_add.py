import os
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initialize CUDA driver context

# Load PTX code from file
ptx_code = '''
.version 5.0
.target sm_50
.address_size 64

.visible .entry vec_add(
    .param .u64 param_a,     // pointer to float* a
    .param .u64 param_b,     // pointer to float* b
    .param .u64 param_c,     // pointer to float* c
    .param .u32 param_n      // int n
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<4>;

    // Load kernel parameters
    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r4,  [param_n];

    // Get thread index = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;

    // Check if idx < n
    setp.lt.s32 %p, %r5, %r4;
    @!%p bra DONE;

    // Compute addresses
    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;

    // Load a[i], b[i], add, store c[i]
    ld.global.f32 %f0, [%rd5];
    ld.global.f32 %f1, [%rd6];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd7], %f2;

DONE:
    ret;
}
'''

# Load module and get kernel function
module = cuda.module_from_buffer(ptx_code.encode())
kernel = module.get_function("vec_add")

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.empty_like(a_gpu)

N = a.shape[0]
block_size = 256
grid_size = (N + block_size - 1) // block_size
kernel(a_gpu, b_gpu, c_gpu, np.uint32(N),
    block=(block_size, 1, 1), grid=(grid_size, 1))

c_gpu.get(c)
