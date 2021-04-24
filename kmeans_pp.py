import argparse
import random
import pandas
import numpy as np
import mykmeanssp as mks

# parser.add_argument('--verbose','-v','--debug','-d', action='count')

num_clusters = 0
num_samples = 0
dim = 0
max_iter = 0
input_filename = ""
centroids = []
centroid_inds = []


def set_consts(c, s, d, m, i, cc, ci):
    global num_clusters, num_samples, dim, max_iter, input_filename, centroids, centroid_inds
    num_clusters = c
    num_samples = c
    dim = d
    max_iter = m
    input_filename = i
    centroid = cc
    centroid_inds = ci


def soft_assert(cond, msg):
    if not cond:
        print(msg)
        exit(1)



RANDOMIZATION_SEED = 0
np.random.seed(RANDOMIZATION_SEED)


def squared_distances(samples, centroids):
    """
        Returns array res of shape (len(samples),len(centroids),)
    for which res[i][j] is the squared distance between sample[i]
    and centroid[j].
    """
    diffs = samples[:, np.newaxis] - centroids
    # diffs[i][j] is samples[i]-centroids[j].
    res = np.sum(diffs ** 2, axis=2)
    return res


def min_squared_distances(samples, centroids):
    """
    Returns array res of shape (len(samples),)
    for which res[i] is the minimal squared distance between sample[i] and
    the centroids.
    """
    res = np.min(squared_distances(samples, centroids), axis=1)
    return res


def weights_to_probs(weights):
    """
    Converts weights to probabilities summing to one
    by dividing each weight by the the weights' total sum.
    """
    return weights / np.sum(weights)


def probs_for_next_centroid_choice(samples, centroids):
    return weights_to_probs(min_squared_distances(samples, centroids))


def choose_new_centroid_ind(samples, centroids):
    return np.random.choice(
        len(samples), p=probs_for_next_centroid_choice(samples, centroids)
    )


# samples = pandas.read_csv(input_filename, header=None).to_numpy()



def add_centroid_by_ind(centroid_ind):
    global centroid_inds, centroids
    centroid_inds.append(centroid_ind)
    centroids = samples[centroid_inds]


def run():
    add_centroid_by_ind(np.random.choice(num_samples))
    for _ in range(num_clusters - 1):
        add_centroid_by_ind(choose_new_centroid_ind(samples, centroids))

    print(",".join(map(str, centroid_inds)))  # print centroid indices
    mks.set_dim(dim)
    x = [a.tolist() for a in centroids]
    mks.set_centroids(x)
    mks.set_samples(samples.tolist())
    mks.iterate(max_iter)
    mks.print_centroids()
