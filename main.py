#!/usr/bin/env python3
import kmeans_numpy as km_np
import algorithms as algs
import random
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt


from kmeans_pp import soft_assert


# TODO measure these
MAX_NUM_CLUSTERS = 4  # 10
MAX_NUM_SAMPLES = 10  # 200


def run(num_clusters, num_samples, is_random):
    try:
        num_clusters = int(num_clusters)
        num_samples = int(num_samples)
    except Exception:
        soft_assert(False, "K,N should be integers")

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

    def jaccard_measure(inds):
        return np.sum(inds == sample_inds) / len(inds)

    with open("data.txt", "w") as f:
        for i, row in enumerate(samples):
            for x in row:
                f.write("%.8f" % x)
                f.write(",")
            f.write("%d" % sample_inds[i])
            f.write("\n")

    spectral_inds, spectral_clusters = algs.norm_spectral_cluster(samples, num_clusters)
    if num_clusters is None:
        num_clusters = len(spectral_inds)
    kmeans_inds, kmeans_clusters = km_np.k_means(samples, num_clusters)

    with open("clusters.txt", "w") as f:
        f.write("%d\n" % num_clusters)
        for inds in [spectral_inds, kmeans_inds]:
            print(inds)
            for i in range(num_clusters):
                f.write(",".join("%d" % j for j, x in enumerate(inds) if x == i))
                f.write("\n")

    # plots
    if dim == 2:
        fig, axes = plt.subplots(1, 2, subplot_kw={"aspect": "equal"})
    else:  # dim==3
        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    for (title, inds, axis) in zip(
        ["Normalized Spectral Clustering", "K-means"], [spectral_inds, kmeans_inds], axes
    ):
        axis.set_title(title)
        if dim == 2:
            axis.scatter(samples[:, 0], samples[:, 1], c=inds)
        else:  # dim==3
            axis.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=inds)
        axis.grid(True, which="both")
        axis.set_xlabel(f"Jaccard measure: {jaccard_measure(inds)}")
    # TODO less ugly text position. y=0.1 looks better but collides with long 2D graphs.
    fig.text(0.1, 0.0, f"Used constants: n = {num_samples}, k = {num_clusters}")
    plt.savefig("clusters.pdf")

    print(f"{num_clusters} clusters with {num_samples} samples, random is {is_random}")
