import numpy as np

from debug_utils import *
dbg = debug_printer(True)

RANDOMIZATION_SEED = 0


def squared_distances(a,b):
    """
        Returns array res of shape (len(a),len(b),)
    for which res[i][j] is the squared distance between a[i]
    and b[j].
    """
    diffs = a[:, np.newaxis] - b
    # diffs[i][j] is samples[i]-centroids[j].
    res = np.sum(diffs ** 2, axis=2)
    return res


def min_squared_distances(a, b):
    """
    Returns array res of shape (len(samples),)
    for which res[i] is the minimal squared distance between sample[i] and
    the centroids.
    """
    res = np.min(squared_distances(a, b), axis=1)
    return res


def weights_to_probs(weights):
    """
    Converts weights to probabilities summing to one
    by dividing each weight by the the weights' total sum.
    """
    return weights / weights.sum()


def _probs_for_next_centroid_choice(samples, centroids):
    return weights_to_probs(min_squared_distances(samples, centroids))


def _choose_new_centroid_ind(samples, centroids):
    return np.random.choice(
        len(samples), p=_probs_for_next_centroid_choice(samples, centroids)
    )

def _k_means_pp_old(samples,k,randomization_seed=0):
    num_samples = len(samples)
    np.random.seed(randomization_seed)
    centroid_inds = []
    centroids = []
    centroid_inds.append(np.random.choice(num_samples))
    centroids = samples[centroid_inds]
    for _ in range(k - 1):
        centroid_inds.append(_choose_new_centroid_ind(samples, centroids))
        centroids = samples[centroid_inds]
    return centroids, centroid_inds

def _k_means_pp(samples, k, randomization_seed=0):
    num_samples = len(samples)
    np.random.seed(randomization_seed)
    centroids_buffer = np.empty((k,samples.shape[1]))
    centroid_inds = np.empty(k,dtype=np.intp)
    centroid_inds[0] = first_ind = np.random.choice(num_samples)
    centroids_buffer[0] = samples[first_ind]
    found_centroids = centroids_buffer[:1]
    for i in range(1, k):
        new_ind = _choose_new_centroid_ind(samples, found_centroids)
        centroid_inds[i] = new_ind
        centroids_buffer[i] = samples[new_ind]
        found_centroids = centroids_buffer[:i+1]
    return centroids_buffer,centroid_inds

# Preform k_means clustering on the given samples, and the given 
# number of clusters k. 

def k_means(samples, k=None, initial_centroids = None, max_iter=300):
    num_clusters = k if k is not None else len(initial_centroids)
    num_samples = len(samples)
    if not 1<=num_clusters<num_samples:
        raise ValueError(f"Illegal number of clusters {num_clusters},"+\
            f"expected number between 1,{num_samples-1}")
    if initial_centroids is None and k is None:
        raise ValueError("Either k or initial_centroids must be supplied")
    elif initial_centroids is None:
        centroids,_ = _k_means_pp(samples,k)
        dbg.print_multiline_vars({'received centroids from kmeans++':centroids})
    else:    
        if k is not None and k!=len(initial_centroids):
            raise ValueError(
                f"Expected k={k} initial centroids, {len(initial_centroids)} were given instead"
                )
        centroids = initial_centroids.copy()
    dbg.d_assert(1<=len(centroids)<num_samples, "")

    # These variables are swapped at the beginning of the loop
    old_s_c_inds = np.empty(num_samples,dtype=np.intp)
    samples_cluster_inds = np.full_like(old_s_c_inds, np.nan)
    for _ in range(max_iter):
        samples_cluster_inds, old_s_c_inds = old_s_c_inds, samples_cluster_inds
        dists = squared_distances(samples,centroids)
        np.argmin(dists,axis=-1,out=samples_cluster_inds)
        #dbg.print_multiline_vars({'samples_cluster_inds':samples_cluster_inds,
            # 'old_inds':old_s_c_inds})
        # There is a one-to-one correspondence between the clusters and the centroids
        # (the clusters are a function of the centroids and the constant samples, and
        # the centroids are a function of the clusters). Thus we can check
        # for change in the clusters instead of in the centroids.

        if np.array_equal(samples_cluster_inds, old_s_c_inds): 
            break
        centroids[:] = 0
        for i,sample in enumerate(samples):
            #dbg.print_vars({'i':i})
            #dbg.print_multiline_vars({'centroids':centroids,f'samples[i={i}]':samples[i],
                # 'sample':sample, 'inds[i]':samples_cluster_inds[i]})
            centroids[samples_cluster_inds[i]] += samples[i]
        cluster_sizes = np.bincount(samples_cluster_inds)
        #dbg.print_multiline_vars({'cluster_sizes':cluster_sizes,'centroids':centroids})
        # assert len(cluster_sizes)==len(centroids) and np.all(cluster_sizes)>=1, \
        #  f"{cluster_sizes}\n{centroids}"
        centroids /= cluster_sizes[:, np.newaxis]
        #dbg.print_vars({"samples[1]":samples[1]})

        #dprint(f"centroids = {centroids}")
    return samples_cluster_inds, centroids

