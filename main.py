#!/usr/bin/env python3
import kmeans_pp
import random
import sklearn.datasets
import numpy as np


from kmeans_pp import soft_assert


# TODO measure these
MAX_NUM_CLUSTERS = 4
MAX_NUM_SAMPLES = 10


def run(num_clusters, num_samples, is_random):
    try:
        num_clusters = int(num_clusters)
        num_samples = int(num_samples)
    except Exception:
        soft_assert(False, "Illegal type for K,N")

    dim = random.randint(2, 3)
    if is_random:
        num_clusters = random.randint(MAX_NUM_CLUSTERS // 2, MAX_NUM_CLUSTERS)
        num_samples = random.randint(MAX_NUM_SAMPLES // 2, MAX_NUM_SAMPLES)

    soft_assert(num_clusters > 0, "K must be a positive integer (bigger than 0)")
    soft_assert(num_samples > 0, "n must be a positive integer (bigger than 0)")
    soft_assert(num_clusters < num_samples, "K must be smaller than N.")
    soft_assert(dim > 0, "d must be a positive integer (bigger than 0).")
    # soft_assert(max_iter >= 0, "MAX_ITER must be a non-negative integer.")

    print(f"Maximum capacity: K={MAX_NUM_CLUSTERS}, N={MAX_NUM_SAMPLES}")

    samples, sample_inds = sklearn.datasets.make_blobs(
        n_samples=num_samples, n_features=dim, centers=num_clusters
    )

    print(sample_inds)
    with open("data.txt", "w") as f:
        for i,row in enumerate(samples):
            for x in row:
                f.write("%.8f" % x)
                f.write(",")
            f.write("%d" % sample_inds[i])
            f.write("\n")

    # should be indecies
    kmeans_clusters = np.zeros([num_clusters, dim])
    spectral_clusters = np.zeros([num_clusters, dim])

    with open("clusters.txt", "w") as f:
        f.write("%d\n" % num_clusters)
        for row in kmeans_clusters:
            f.write(",".join("%d" % x for x in row))
            f.write("\n")
        for row in spectral_clusters:
            f.write(",".join("%d" % x for x in row))
            f.write("\n")

    print(f"{num_clusters} clusters with {num_samples} samples, random is {is_random}")
