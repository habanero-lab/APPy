# Number of threads assigned per parallel loop iteration when SIMD inner loops
# are detected. Matches the hardwired maxTotalThreadsPerThreadgroup used by
# metalcompute on all Apple Silicon GPUs (M1 through M4).
SIMD_WIDTH = 1024
