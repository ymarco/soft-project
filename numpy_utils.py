"""
Module for general utility functions on numpy arrays.
"""

import numpy as np

def weights_to_probs(weights):
    """
    Converts weights to probabilities summing to one
    by dividing each weight by the the weights' total sum.
    """
    return weights / weights.sum()

def diff_mat(a, b):
    """
    For the return value ret, ret[i,j] is equal to a[i]-b[j].
    """
    return a[:, np.newaxis] - b

def squared_distances(a,b):
    """
    Returns the squared distances between vectors in a and vectors in b:
    the return value ret is of shape (len(a),len(b),)
    for which ret[i,j] is the squared distance between a[i]
    and b[j].
    """
    diffs = diff_mat(a,b)
    res = np.sum(diffs ** 2, axis=-1)
    return res

def euc_dist_mat(a, b):
    """
    Returns eucledian distance ret of shape (len(a),len(b),)
    for which ret[i,j] is the eucledian distance between a[i]
    and b[j].
    """
    return np.linalg.norm(diff_mat(a, b), axis=-1)

def row_sums(mat):
    """
    Returns an array containing the sums across last axis
    (sum of each row for our case, given a matrix).
    """
    return np.sum(mat, axis=-1)

def row_norms(mat):
    """
    Returns an array containing the norms across last axis
    (norm of each row for our case, given a matrix).
    """
    return np.linalg.norm(mat,axis=-1)

def normalize_rows(mat):
    """
    Normalize every row in mat.
    """
    mat /= row_norms(mat)[:,np.newaxis]

def all_rows_nonzero(mat):
    return np.all(np.any(mat!=0,axis=-1))

def argsort_k_smallest(arr, k):
    """
    Returns the indices which yield the first k smallest elements in the
    given array, in sorted order. (Equivalent to np.argsort(arr)[:k])
    """

    # Using argpartition and sorting only the k relevant indices
    # yields better performance than plain sorting (commented),
    # this is backed up by benchmarking.
    #np.argsort(arr)[:k]
    smallest_k_partition_inds = np.argpartition(arr,k)[:k]
    partition_sorting_inds = np.argsort(arr[smallest_k_partition_inds])
    return smallest_k_partition_inds[partition_sorting_inds]
