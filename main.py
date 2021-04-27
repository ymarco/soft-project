#!/usr/bin/env python3
"""
This file is the main program, testing the kmeans and spectral clustering
algorithms, and performing the visualization.
"""

import random
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import itertools
from basic_utils import err_exit_with, arg_to_int, soft_assert
from clustering_algs import norm_spectral_clustering, k_means

# TODO measure these
MAX_CAP_SAMPLES_2 = 301
MAX_CAP_CLUSTERS_2 = 32
MAX_CAP_SAMPLES_3 = 300
MAX_CAP_CLUSTERS_3 = 30

def same_cluster_bool_arr(inds):
    """
    Returns an array ret which contains truth values for indices
    corresponding to pairs i<j of sample indices only when samples i,j
    are in the same cluster according to the sample to cluster index
    mapping in parameter inds.
    """
    return [i==j for i,j in itertools.combinations(inds,2)]

def jaccard_measure(inds1, inds2):
    arr1 = same_cluster_bool_arr(inds1)
    arr2 = same_cluster_bool_arr(inds2)
    same_cluster_pairs_both = \
        np.count_nonzero(np.logical_and(arr1,arr2))
    total_same_cluster_pairs = \
        np.count_nonzero(arr1)+np.count_nonzero(arr2)
    return same_cluster_pairs_both/total_same_cluster_pairs

def write_cluster_members(f,inds):
    memb_lists = cluster_member_i_list(inds)
    f.write('\n'.join(
        ','.join(map(str,l)) for l in memb_lists
    ) + '\n')

def save_computed_clusters(out_file_name, num_search_clusters, *output_ind_lists):
    with open(out_file_name, "w") as f:
        f.write("%d\n" % num_search_clusters)
        for inds in output_ind_lists:
            write_cluster_members(f,inds)

def cluster_member_i_list(inds):
    """
    Returns a list ret for which ret[i] is the list
    of sample indices belonging to the cluser i according
    to the sample to cluster index mappings in parameter inds.
    """
    res = [[] for _ in range(max(inds)+1)]
    for samp_i,cluster_i in enumerate(inds):
        res[cluster_i].append(samp_i)
    return res

def save_gen_data(out_file_name, samples,cluster_inds):
    with open(out_file_name, "w") as f:
        for i, row in enumerate(samples):
            for x in row:
                f.write(f"{x:.8f},")
            f.write(f"{cluster_inds[i]}\n")


def get_settings_from_args(k, n, is_random, dim):
    """
    Parses the arguments and returns:
    1. The number of samples to generate
    and work with.
    2. The dimension of the samples.
    3. The maximum capacity for n matching the dimension.
    4. the maximum capacity for k matching the dimension.
    5. The number of clusters in the generation.
    6. The number of clusters to find using the algorithms,
       or None if the eigengap heuristic should be used.
    """
    if dim is None:
        dim = random.randint(2, 3)
    else:
        dim = arg_to_int('dim', dim)

    max_cap_samples = (MAX_CAP_SAMPLES_2,MAX_CAP_SAMPLES_3)[dim-2]
    max_cap_clusters = (MAX_CAP_CLUSTERS_2,MAX_CAP_CLUSTERS_3)[dim-2]
    if is_random:
        min_n = max_cap_samples // 2
        max_n = max_cap_samples

        min_k = max_cap_clusters // 2
        max_k = max_cap_clusters

        # Using max(min_k+1,min_n) for n ensures we don't get n=min_k
        # which makes it impossible to choose k<n.
        # Using min(max_k,n-1) for k is also required to ensure that we
        # choose k<n.
        n = random.randint(max(min_k+1,min_n), max_n)
        k = random.randint(min_k, min(max_k,n-1))
        num_search_clusters = None
    else:
        if k is None or n is None:
            err_exit_with(
                "missing argument k or n (must be suppplied\ "
                "with --no-Random)")
        k = arg_to_int('k', k)
        n = arg_to_int('n', n)
        num_search_clusters = k

    num_samples = n
    num_gen_clusters = k
    #soft_assert(dim in (2,3), "dim must be 2 or 3.")
    soft_assert(k > 0, "argument k must be a positive integer (bigger than 0)")
    soft_assert(n > 0, "argument n must be a positive integer (bigger than 0)")
    soft_assert(k < n, f"argument k={k} must be smaller than n.")
    return num_samples, dim, max_cap_samples, max_cap_clusters,\
        num_gen_clusters, num_search_clusters

def create_plots(
        out_pdf_file_name,
        samples, num_gen_clusters, num_search_clusters,
        gen_inds, spectral_inds, k_means_inds,
):
    """
    Create the visual plots and output them to out_pdf_file_name_name
    """
    num_samples, dim = samples.shape
    if dim == 2:
        fig, axes = plt.subplots(1, 2, subplot_kw={"aspect": "equal"})
    else:  # dim==3
        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    for (title, inds, axis) in zip(
        ("Normalized Spectral Clustering", "K-means"),
        (spectral_inds, k_means_inds),
        axes,
    ):
        axis.set_title(title)
        if dim == 2:
            axis.scatter(samples[:, 0], samples[:, 1], c=inds)
        else:  # dim==3
            axis.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=inds)
        axis.grid(True, which="both")
        jacc_measure = jaccard_measure(inds, gen_inds)
        axis.set_xlabel(f"Jaccard measure: {jacc_measure:.4f}")

    fig.text(0.033, 0.05,
             f"Used n = {num_samples}, k = {num_gen_clusters} for data "
             f"generation, searched for k = {num_search_clusters} "
             f"clusters in the algorithms. ")
    plt.savefig(out_pdf_file_name)

def run(k, n, is_random, dim):
    """Main function that is called from tasks.py."""

    num_samples, dim, max_cap_samples, max_cap_clusters,\
        num_gen_clusters, num_search_clusters = get_settings_from_args(
            k, n, is_random, dim)

    print(
        f"Running with n={num_samples}, dim={dim}. K={num_gen_clusters} "
        f"for generation and k={num_search_clusters} for the clustering "
        f"algorithms.")
    print(f"Maximum capacity: k={max_cap_clusters}, n={max_cap_samples}")

    samples, gen_inds = sklearn.datasets.make_blobs(
        n_samples=num_samples, n_features=dim, centers=num_gen_clusters
    )
    save_gen_data("data.txt", samples, gen_inds)

    spectral_inds, spectral_clusters = norm_spectral_clustering(
        samples, num_search_clusters
    )

    if num_search_clusters is None:
        num_search_clusters = len(spectral_clusters)
    k_means_inds, k_means_clusters = k_means(
        samples, num_search_clusters
    )

    save_computed_clusters(
        "clusters.txt",
        num_search_clusters, spectral_inds, k_means_inds,
    )

    create_plots(
        "clusters.pdf",
        samples, num_gen_clusters, num_search_clusters,
        gen_inds, spectral_inds, k_means_inds,
    )
