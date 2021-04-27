#!/usr/bin/env python3
import random
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
from basic_utils import err_exit_with, arg_to_int, soft_assert
from clustering_algs import norm_spectral_clustering, k_means

# TODO measure these
MAX_NUM_CLUSTERS = 10
MAX_NUM_SAMPLES = 200



def run(k, n, is_random):
    """Main function that is called from tasks.py."""

    if is_random:
        min_n = MAX_NUM_SAMPLES // 2
        max_n = MAX_NUM_SAMPLES
        min_k = MAX_NUM_CLUSTERS // 2
        max_k = MAX_NUM_CLUSTERS

        # Using max(min_k+1,min_n) for n ensures we don't get n=min_k
        # without possibilities
        # Using min(max_k,n-1) for k is required to ensure that we
        # choose k<n.
        n = random.randint(max(min_k+1,min_n), max_n)
        k = random.randint(min_k, min(max_k,n-1))
        num_search_clusters = None
    else:
        if k is None or n is None:
            err_exit_with(
                "missing argument k or n (must be suppplied\ with --no-Random)")
        k = arg_to_int('k', k)
        n = arg_to_int('n', n)
        num_search_clusters = k

    num_samples = n
    num_gen_clusters = k
    dim = random.randint(2, 3)

    print(k,num_gen_clusters,num_search_clusters)
    print(n)

    soft_assert(k > 0, "argument k must be a positive integer (bigger than 0)")
    soft_assert(n > 0, "argument n must be a positive integer (bigger than 0)")
    soft_assert(k < n, f"argument k={k} must be smaller than n.")

    print(f"Maximum capacity: k={MAX_NUM_CLUSTERS}, n={MAX_NUM_SAMPLES}")
    print(f"Running with k={k}, n={n}")

    samples, sample_inds = sklearn.datasets.make_blobs(
        n_samples=num_samples, n_features=dim, centers=num_gen_clusters
    )

    def cluster_set(i, inds):
        """Returns a set with all of sample i's neighboors in the cluster."""
        return { j for j in range(len(samples)) if sample_inds[i] == inds[j] }

    def jaccard_measure(inds):
        """See project specs."""
        sum_ = 0
        for i, sample in enumerate(samples):
            sum_ += (cluster_set(i, sample_inds) == cluster_set(i, inds))
        return sum_ / num_samples

    with open("data.txt", "w") as f:
        for i, row in enumerate(samples):
            for x in row:
                f.write("%.8f" % x)
                f.write(",")
            f.write("%d" % sample_inds[i])
            f.write("\n")

    spectral_inds, spectral_clusters = norm_spectral_clustering(samples, num_search_clusters)
    if num_search_clusters is None:
        num_search_clusters = len(spectral_clusters)
    k_means_inds, k_means_clusters = k_means(samples, num_search_clusters)

    with open("clusters.txt", "w") as f:
        f.write("%d\n" % num_gen_clusters)
        for inds in [spectral_inds, k_means_inds]:
            # print(inds)
            for i in range(num_gen_clusters):
                f.write(",".join("%d" % j for j, x in enumerate(inds) if x == i))
                f.write("\n")

    # plots
    if dim == 2:
        fig, axes = plt.subplots(1, 2, subplot_kw={"aspect": "equal"})
    else:  # dim==3
        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    for (title, inds, axis) in zip(
        ["Normalized Spectral Clustering", "K-means"],
        [spectral_inds, k_means_inds],
        axes,
    ):
        axis.set_title(title)
        if dim == 2:
            axis.scatter(samples[:, 0], samples[:, 1], c=inds)
        else:  # dim==3
            axis.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=inds)
        axis.grid(True, which="both")
        axis.set_xlabel(f"Jaccard measure: {jaccard_measure(inds):.4f}")
    # TODO less ugly text position. y=0.1 looks better but collides with long 2D graphs.
    fig.text(0.1, 0.0, f"Used constants: n = {num_samples}, k = {num_gen_clusters}")
    plt.savefig("clusters.pdf")
