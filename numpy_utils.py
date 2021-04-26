"""
Module for basic useful utility functions on numpy arrays
"""

import numpy as np

def weights_to_probs(weights):
    """
    Converts weights to probabilities summing to one
    by dividing each weight by the the weights' total sum.
    """
    return weights / weights.sum()

def diff_mat(a, b):
    return a[:, np.newaxis] - b

def squared_distances(a,b):
    """
    Returns the squared distances between vectors in a and vectors in b:
    the return value ret is of shape (len(a),len(b),)
    for which ret[i][j] is the squared distance between a[i]
    and b[j].
    """
    diffs = diff_mat(a,b)
    # diffs[i][j] is samples[i]-centroids[j].
    res = np.sum(diffs ** 2, axis=-1)
    return res

def euc_dist_mat(a, b):
    """
    Returns eucledian distance ret of shape (len(a),len(b),)
    for which ret[i][j] is the eucledian distance between a[i]
    and b[j].
    """
    return np.linalg.norm(diff_mat(a, b), axis=-1)

def row_sums(mat):
    return np.sum(mat, axis=0)

def row_norms(mat):
    return np.linalg.norm(mat,axis=-1)

def normalize_rows(mat):
   mat /= row_norms(mat)[:,np.newaxis]

def all_rows_nonzero(mat):
    return np.all(np.any(mat!=0,axis=-1))


# Returns the indices which yield the first k smallest elements in the
# given array, in sorted order. (Equivalent to np.argsort(arr)[:k])
def argsort_k_smallest(arr, k):
    #TODO: consider using:
    #smallest_k_partition_inds = np.argpartition(arr,k)[:k]
    #partition_sorting_inds = np.argsort(arr[smallest_k_partition_inds])
    #return smallest_k_partition_inds[partition_sorting_inds]
    return np.argsort(arr)[:k]
