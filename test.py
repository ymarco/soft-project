#!/usr/bin/env python3


import sklearn.datasets, numpy as np

#A = np.zeros((4,4))
#print(qr_decomposition_destructive(A))

from algorithms import *

from debug_utils import *
set_debug(True)
#samples = np.array([[0,1],[2,3]])
samples, cluster_labels = sklearn.datasets.make_blobs(3,2)
print_vars({'samples':samples})
u = norm_spectral_cluster(samples)
print(u)
