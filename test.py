#!/usr/bin/env python3


import sklearn.datasets, numpy as np

#A = np.zeros((4,4))
#print(qr_decomposition_destructive(A))

from algorithms import *

import debug_utils
dbg = debug_utils.debug_printer(True)
#samples = np.array([[0,1],[2,3]])
NUM_SAMPLES = 5
samples, cluster_labels = sklearn.datasets.make_blobs(NUM_SAMPLES,2)
dbg.print_multiline_vars({'samples':samples})

# I = np.identity(3)
# qr_decomposition_destructive(I)

u = norm_spectral_cluster(samples)
dbg.print_vars({'u':u})
