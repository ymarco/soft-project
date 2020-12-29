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

RANDOMIZATION_SEED = 0
np.random.seed(RANDOMIZATION_SEED)
samples = pandas.read_csv(input_filename, header=None).to_numpy()
centroid_inds = [np.random.choice(num_samples)]
for _ in range(num_clusters - 1):
    centroids = samples[centroid_inds]
    diffs = samples[:,np.newaxis]-centroids # diffs[i][j] is samples[i]-centroids[j].
    squared_distances = np.sum(diffs**2,axis=2) 
    min_squared_distances = np.min(squared_distances,axis=1)
    probs = min_squared_distances/np.sum(min_squared_distances)
    centroid_inds.append(np.random.choice(num_samples, p=probs))
centroids = samples[centroid_inds]

print(",".join(map(str, centroid_inds)))
mks.set_dim(dim)
x = [a.tolist() for a in centroids]
mks.set_centroids(x)
mks.set_samples(samples.tolist())
mks.iterate(max_iter)
mks.print_centroids()
