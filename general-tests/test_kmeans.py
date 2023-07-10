import numpy as np
import torch
from sklearn.cluster import KMeans
from slap.utils import bench

torch.set_default_device('cuda')

class GPUKMeans():
    def __init__(self, n_clusters, init, max_iter):
        self.n_clusters = n_clusters
        self.init = torch.from_numpy(init).to(torch.float32).to('cuda')
        self.centers = self.init
        self.max_iter = max_iter

    def fit(self, x):
        x = torch.from_numpy(x).to(torch.float32).to('cuda')
        for _ in range(self.max_iter):
            dists = torch.cdist(x, self.centers, compute_mode='use_mm_for_euclid_dist')
            labels = torch.argmin(dists, axis=-1)

            for i,c in enumerate(self.centers):
                selected = x[labels == i]
                self.centers[i] = torch.mean(selected, axis=0)

        dists = torch.cdist(x, self.centers)
        labels = torch.argmin(dists, axis=-1)
        return labels

np.random.seed(0)

M = 60000
N = 512
C = 128
X = np.random.randn(M, N).astype(np.float32)

init_centers = np.random.randn(C, N).astype(np.float32)
init_centers1 = init_centers.copy()
max_iters = 30
kmeans = KMeans(n_clusters=C, random_state=0, n_init=1, init=init_centers, max_iter=max_iters)
kmeans.fit(X)

labels_cpu = kmeans.labels_
print('kmeans labels:', labels_cpu)
print('kmeans sklearn:', bench(lambda: kmeans.fit(X))/kmeans.n_iter_)

gkmeans = GPUKMeans(n_clusters=C, init=init_centers1, max_iter=max_iters)
labels_gpu = gkmeans.fit(X)
print(labels_gpu)
print(np.sum(labels_cpu != labels_gpu.cpu().numpy()))
print(bench(lambda: gkmeans.fit(X))/max_iters)

                                                
