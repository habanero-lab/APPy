import numpy as np
import numba
import appy
import appy.np_shared as nps
from time import perf_counter


# -----------------------------------------------------
# NumPy version (vectorized but simple, for correctness)
# -----------------------------------------------------
def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter):
    xs = np.linspace(xmin, xmax, width, dtype=np.float32)
    ys = np.linspace(ymin, ymax, height, dtype=np.float32)
    out = np.zeros((height, width), dtype=np.int32)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            c_re = x
            c_im = y
            z_re = np.float32(0.0)
            z_im = np.float32(0.0)
            it = 0

            while z_re * z_re + z_im * z_im <= np.float32(4.0) and it < max_iter:
                # z = z*z + c
                new_re = z_re * z_re - z_im * z_im + c_re
                new_im = np.float32(2.0) * z_re * z_im + c_im
                z_re = new_re
                z_im = new_im
                it += 1

            out[i, j] = it
    return out


# -----------------------------------------------------
# Numba version
# -----------------------------------------------------
@numba.njit(parallel=True)
def mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter, out):
    for i in numba.prange(height):
        y = ymin + (ymax - ymin) * i / (height - 1)
        for j in range(width):
            x = xmin + (xmax - xmin) * j / (width - 1)

            c_re = x
            c_im = y
            z_re = np.float32(0.0)
            z_im = np.float32(0.0)
            it = 0

            while z_re * z_re + z_im * z_im <= np.float32(4.0) and it < max_iter:
                new_re = z_re * z_re - z_im * z_im + c_re
                new_im = np.float32(2.0) * z_re * z_im + c_im
                z_re = new_re
                z_im = new_im
                it += 1

            out[i, j] = it


# -----------------------------------------------------
# APPy version
# -----------------------------------------------------
@appy.jit(verbose_static_rewrite=True, dump_code=True)
def mandelbrot_appy(xmin, xmax, ymin, ymax, width, height, max_iter, out):
    size = width * height
    for idx in appy.prange(size):
        i = idx // width
        j = idx % width

        y = ymin + (ymax - ymin) * i / (height - 1)
        x = xmin + (xmax - xmin) * j / (width - 1)

        c_re = x
        c_im = y
        z_re = 0.0
        z_im = 0.0
        it = 0

        while z_re * z_re + z_im * z_im <= 4.0 and it < max_iter:
            new_re = z_re * z_re - z_im * z_im + c_re
            new_im = 2.0 * z_re * z_im + c_im
            z_re = new_re
            z_im = new_im
            it += 1

        out[idx] = it



# -----------------------------------------------------
# Test harness
# -----------------------------------------------------
def test_mandelbrot():
    width = 1500
    height = 1000
    max_iter = 22
    xmin, xmax = np.float32(-2.0), np.float32(1.0)
    ymin, ymax = np.float32(-1.5), np.float32(1.5)

    # Output buffers
    out_appy  = nps.zeros((height, width), dtype=np.int32)
    out_numba = np.zeros((height, width), dtype=np.int32)
    out_np    = mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter)

    # Warmup    
    out_appy.arr = out_appy.arr.reshape(-1)
    mandelbrot_appy(xmin, xmax, ymin, ymax, width, height, max_iter, out_appy)
    out_appy.arr = out_appy.arr.reshape(height, width)
    mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter, out_numba)


    # exit(1)
    # Correctness checks
    assert np.allclose(out_np, out_appy, atol=0)
    #assert np.allclose(out_np, out_numba, atol=0)

    # Timing NumPy
    t0 = perf_counter()
    out_np = mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter)
    t1 = perf_counter()
    print(f"NumPy:  {1000*(t1-t0):.2f} ms")

    # Timing APPy
    t0 = perf_counter()
    out_appy.arr = out_appy.arr.reshape(-1)
    mandelbrot_appy(xmin, xmax, ymin, ymax, width, height, max_iter, out_appy)
    out_appy.arr = out_appy.arr.reshape(height, width)
    t1 = perf_counter()
    print(f"APPy:   {1000*(t1-t0):.2f} ms")

    # Timing Numba
    t0 = perf_counter()
    mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter, out_numba)
    t1 = perf_counter()
    print(f"Numba:  {1000*(t1-t0):.2f} ms")


if __name__ == "__main__":
    test_mandelbrot()
