import argparse, random, pandas, numpy as np
import mykmeanssp as mks

parser = argparse.ArgumentParser()
parser.add_argument("K", type=int)
parser.add_argument("N", type=int)
parser.add_argument("d", type=int)
parser.add_argument("MAX_ITER", type=int)
parser.add_argument("filename", type=str)

# parser.add_argument('--verbose','-v','--debug','-d', action='count')

args = parser.parse_args()
num_clusters = args.K
num_samples = args.N
dim = args.d
max_iter = args.MAX_ITER
input_filename = args.filename


def soft_assert(cond, msg):
    if not cond:
        print(msg)
        exit(1)


soft_assert(num_clusters > 0, "K must be a positive integer (bigger than 0)")
soft_assert(num_clusters < num_samples, "K must be smaller than N.")
soft_assert(dim > 0, "d must be a positive integer (bigger than 0).")
soft_assert(max_iter >= 0, "MAX_ITER must be a non-negative integer.")


def squared_distances(vec, vecs):
    return np.sum((vecs - vec) ** 2, axis=1)


def min_squared_distance(vec, vecs):
    return np.amin(squared_distances(vec, vecs))


RANDOMIZATION_SEED = 0
np.random.seed(RANDOMIZATION_SEED)
samples = pandas.read_csv(input_filename, header=None).to_numpy()
centroid_inds = [np.random.choice(num_samples)]
for _ in range(num_clusters - 1):
    centroids = np.array([samples[i] for i in centroid_inds])
    weights = np.fromiter(
        (min_squared_distance(sample, centroids) for sample in samples), float
    )
    weights /= np.sum(weights)
    centroid_inds.append(np.random.choice(num_samples, p=weights))

print(",".join(map(str, centroid_inds)))
mks.set_dim(dim)
x = [a.tolist() for a in centroids]
mks.set_centroids(x)
mks.set_samples(samples.tolist())
mks.iterate(max_iter)
mks.print_centroids()
