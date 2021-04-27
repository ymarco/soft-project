#!/usr/bin/env python3


import sklearn.datasets, numpy as np

from numpy_utils import *
a = np.random.rand(50000)
v1 = argsort_k_smallest(a,1234)
v2 = argsort_k_smallest_2(a,1234)
print(np.all(v1==v2))
print(a[:5])

import timeit
print(timeit.timeit("argsort_k_smallest(a,1234)",setup=
"""from numpy_utils import argsort_k_smallest
import numpy as np
np.random.seed(0)
a = np.random.rand(50000)""",number=200))

print(timeit.timeit("argsort_k_smallest_2(a,1234)",setup=
"""from numpy_utils import argsort_k_smallest_2
import numpy as np
np.random.seed(0)
a = np.random.rand(50000)""",number=200))

#A = np.zeros((4,4))
#print(qr_decomposition_destructive(A))

#from clustering_algs import *
#
#import debug_utils
#dbg = debug_utils.debug_printer(True)
##samples = np.array([[0,1],[2,3]])
#
#
#NUM_SAMPLES = 6
#samples, cluster_labels = sklearn.datasets.make_blobs(NUM_SAMPLES,2)
#dbg.print_multiline_vars({'samples':samples})
#
## I = np.identity(3)
## qr_decomposition_destructive(I)
#
#u = norm_spectral_clustering(samples)
#dbg.print_multiline_vars({'u':u})
#
## from kmeans_numpy import *
## samples = np.array([[0.,1.,2.],[1.,15.,12.],[2.,18.,17.],[-2.,3.,4.],[-5.,6.,10.]])
## k = 3
## dbg.print_vars({
## 	'k_means++ output':k_means_pp(samples,k),
## 	'old k_means++ output':k_means_pp_old(samples,k)
## 	})
#
## res = k_means(samples,k,max_iter=1000)
## # res = k_means(samples,initial_centroids=samples[:k],max_iter=1000)
## inds, centroids = res
## print(f"inds = {inds}\n\ncentroids:\n{centroids}")
