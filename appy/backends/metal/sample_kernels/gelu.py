
# GELU kernel in Metal (approximate version)
kernel_str = """
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
    const device float* x [[ buffer(0) ]],
    device float* y [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]])
{
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float xi = x[id];
    float c = 0.044715f;
    float sqrt_2_over_pi = 0.79788456f;
    float x3 = xi * xi * xi;
    float t = tanh(sqrt_2_over_pi * (xi + c * x3));
    y[id] = 0.5f * xi * (1.0f + t);
}
"""


def kernel_loop_1(x, x_shape_0, y):
    if not hasattr(kernel_loop_1, "kernel"):
        kernel_loop_1.kernel = x.dev.kernel(kernel_str).function("gelu")
    handle = kernel_loop_1.kernel(x_shape_0, x.buf, y.buf)
    del handle
