import argparse, random, pandas, numpy

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
    return numpy.sum((vecs - vec) ** 2, axis=1)


def min_squared_distance(vec, vecs):
    return numpy.amin(squared_distances(vec, vecs))


RANDOMIZATION_SEED = 0
random.seed(RANDOMIZATION_SEED)
samples = pandas.read_csv(input_filename, header=None).to_numpy()
centriods = [samples[numpy.random.choice(num_samples)]]
for _ in range(num_clusters - 1):
    weights = numpy.fromiter(
        (min_squared_distance(sample, centriods) for sample in samples), float
    )
    weights /= numpy.sum(weights)
    centriods.append(samples[numpy.random.choice(num_samples, p=weights)])

print(centriods)
