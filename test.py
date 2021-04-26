#!/usr/bin/env python3


import sklearn.datasets, numpy as np

#A = np.zeros((4,4))
#print(qr_decomposition_destructive(A))

from algorithms import *

import debug_utils
dbg = debug_utils.debug_printer(True)
#samples = np.array([[0,1],[2,3]])
NUM_SAMPLES = 4
samples, cluster_labels = sklearn.datasets.make_blobs(NUM_SAMPLES,2)
dbg.print_multiline_vars({'samples':samples})

# I = np.identity(3)
# qr_decomposition_destructive(I)

u = norm_spectral_cluster(samples)
dbg.print_multiline_vars({'u':u})

from kmeans_numpy import *
samples = np.array([[0.,1.,2.],[1.,15.,12.],[2.,18.,17.],[-2.,3.,4.],[-5.,6.,10.]])
k = 3
res = k_means(samples,initial_centroids=samples[:k],max_iter=1000)
inds, centroids = res
print(f"inds = {inds}\n\ncentroids:\n{centroids}")
