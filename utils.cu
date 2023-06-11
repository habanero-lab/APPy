#define FULL_MASK 0xffffffff

__inline__ __device__ float blockReduceSum(float sum) {
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    for (int step = 32/2; step > 0; step = step / 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, step);
    }

    // Aggregate across warps
    __shared__ float psums[32]; // Max number of warps per block
    if (laneId == 0) {
        psums[warpId] = sum;
    }
    __syncthreads();  // Wait for all warps to finish

    if (warpId == 0) {
        // `psum` may only be partially used. 
        // For example, if blockDim.x == 256, then only the first 8 elements 
        // actually contain the partial sum in `psum`
        if (laneId < blockDim.x / 32) {
            sum = psums[laneId];
        }
        else {
            sum = 0;
        }
        
        for (int step = 32/2; step > 0; step = step / 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, step);
        }

        if (laneId == 0) {
            psums[0] = sum;
        }
    }
    __syncthreads();   // make sure psums[0] has already been written

    return psums[0];
}