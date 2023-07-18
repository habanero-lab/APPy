import torch
import appy

def kernel(x, labels, centers):
    '''
    In this version, when the number of clusters is small, like 4 or 8, because
    each program instance is not doing much work, parallelism could be limited.
    Writing to the same location requires atomic operation is a sequential process.
    '''
    for i in range(x.shape[0]):  #pragma parallel reduction(+:centers)
        for j in range(0, x.shape[1], Bj):  #pragma parallel
            label = labels[i]
            centers[label,j:j+Bj] += x[i,j:j+Bj]
            
def kernel1(x, labels, centers, M, N, BM, BN, nclusters):
    '''
    This version uses a local storage for each program instance, and tries to 
    first accumulate in the local storage, before writing the final centers.
    This could lead to more parallelism in terms of reducing serialization.
    However it does use more on-chip resource (nclusters*BN array), which could 
    in turn limit parallelism.
    '''
    for i in range(0, M, BM):  #pragma parallel reduction(+:centers)
        for j in range(0, x.shape[1], BN):  #pragma parallel
            local_centers = torch.zeros([nclusters, BN])
            for ii in range(i, i+BM):
                label = labels[ii]
                local_centers[label,j:j+Bj] += x[i,j:j+Bj]
            
            for k in range(nclusters):
                centers[k,j:j+Bj] += local_centers[k,j+Bj]

def kernel2(x, labels, centers, M, N, BM, BN, nclusters):
    '''
    This version eliminates parallel reduction and the serialization by letting 
    each program instance scan a designated cluster id. This will iterate `x` many
    times and proably will be quite inefficient, but it's still a valid algorithm.
    '''
    for k in range(nclusters):  #pragma parallel
        local_centers = torch.zeros([BN])
        for i in range(M):
            for j in range(0, x.shape[1], BN):  #pragma parallel
                label = labels[i]
                if k == label:
                    local_centers[j:j+Bj] += x[i,j:j+Bj]
        centers[k,j:j+Bj] += local_centers[j:j+Bj]
