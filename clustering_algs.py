import numpy as np
import debug_utils
dbg = debug_utils.debug_printer(False)

import numpy_utils as np_utils
import mat_algs

EPSILON = 0.0001

def _min_squared_distances(samples, centroids):
    """
    Returns array res of shape (len(samples),)
    for which res[i] is the minimal squared distance between sample[i] and
    the centroids.
    """
    res = np.min(np_utils.squared_distances(samples, centroids), axis=-1)
    return res


def _probs_for_next_centroid_choice(samples, centroids):
    return np_utils.weights_to_probs(_min_squared_distances(samples, centroids))


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
            f"expected number between 1, {num_samples-1}")
    if initial_centroids is None and k is None:
        raise ValueError("Either k or initial_centroids must be supplied")
    elif initial_centroids is None:
        centroids,_ = _k_means_pp(samples,k)
        # dbg.print_multiline_vars({'received centroids from kmeans++':centroids})
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
        dists = np_utils.squared_distances(samples,centroids)
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


def _weight_adj_mat(samples):
    """
    Calculate the weight adjacency matrix.
    The resulting matrix ret is a symmetric matrix with diagonal of 0,
    (ret[i][i]=0) and for i!=j, ret[i][j] is equal to:
        exp(-l2_norm(samples[i]-samples[j])/2)
    """

    # TODO: optimize this by avoiding calculation of symmetric elements
    # twice and avoiding calculation for points of the same index
    # (which we know are zeros on the diagonal).

    dists = np_utils.euc_dist_mat(samples, samples)
    res = np.exp(-dists / 2)
    np.fill_diagonal(res, 0)

    return res

# 3.2, 3.3
def _norm_graph_lap(samples):
    weights = _weight_adj_mat(samples)

    # In these comments we will denote weighted adjacency matrix as W
    # (as in step 3.1 in the algorithm) and the
    # the diagonal degree matrix as D
    # (as in step 3.2)

    # This is the diagonal of D as shown in step 3.1
    rsqrt_row_sums = np_utils.row_sums(weights) ** (-0.5)

    # This is the calculation of D**(-0.5)@W@D**(-0.5) with D being the diagonal
    # degree matrix from step 3.3.
    # In order to reduce space and performance costs, we can avoid
    # calculating and storing the actual diagonal matrix D with useless
    # zeroes. We do that by multiplying the columns of W by D's diagonal
    # and then multiplying the result's rows by the diagonal. Since D is
    # a diagonal matrix this is exactly the same as performing the matrix
    # multiplication.
    weights *= rsqrt_row_sums
    weights *= rsqrt_row_sums[:, np.newaxis]

    prod = weights
    # We've calculated D**(-0.5)@W@D**(-0.5), all that's left
    # is to subtract 1 from the diagonal and negate the array
    np.fill_diagonal(prod,(np.diag(weights)-1))
    return -prod


# If k==None, uses the eigengap heuristic
def _k_smallest_eigenvalue_inds(e_vals, k=None):
    if k is not None:
        return np_utils.argsort_k_smallest(e_vals,k)
    e_vals_first_half_sort_inds = np_utils.argsort_k_smallest(e_vals, len(e_vals)//2+1)
    e_vals_sorted_first_half = e_vals[e_vals_first_half_sort_inds]
    e_gaps = np.abs(np.diff(e_vals_sorted_first_half))
    dbg.print_vars({'e_vals_sorted_first_half':e_vals_sorted_first_half
        ,'e_gaps':e_gaps})
    k = np.argmax(e_gaps)+1
    dbg.print_vars(('k',k))
    return e_vals_first_half_sort_inds[:k]

def norm_spectral_clustering(samples, k=None):
    dbg = debug_utils.debug_printer(False)
    dim = len(samples)
    l = _norm_graph_lap(samples)

    # TODO: rename e_val_mat to e_val_c_mat and e_vec_mat to e_vec_d_mat
    # (for columns and diagonal, respectively)

    # TODO: check if adding a bit to the diagonal yeilds better clusters
    PRE_QR_ITERATION_ADDEND = 0 if True else (2*EPSILON)*np.identity(len(l))
    l+=PRE_QR_ITERATION_ADDEND
    e_val_mat,e_vec_mat = mat_algs.qr_iteration(l,epsilon=EPSILON)
    e_vals = e_val_mat.diagonal()
    dbg.d_assert(np.all(e_vals>=-EPSILON),
        "Encountered negative eigenvalues, "
        + debug_utils.vars_to_str(('e_vals',e_vals)))
    dbg.print_vars({'e_val_mat':e_val_mat, 'e_vals':e_vals, 'e_vec_mat':e_vec_mat})
    inds = _k_smallest_eigenvalue_inds(e_vals,k)
    u = e_vec_mat[:,inds]
    if not np_utils.all_rows_nonzero(u):
        raise RuntimeError("U in the spectral clustering algorithm has a zero row, can not be normalized.")
    np_utils.normalize_rows(u)
    return k_means(u,u.shape[1],max_iter=300)
