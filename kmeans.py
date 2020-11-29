import argparse
parser = argparse.ArgumentParser()
parser.add_argument('K', type=int)
parser.add_argument('N', type=int)
parser.add_argument('d', type=int)
parser.add_argument('MAX_ITER', type=int)
args = parser.parse_args()
num_clusters=args.K
num_samples=args.N
dim=args.d
max_iter=args.MAX_ITER
#print(num_clusters, num_samples, dim)

def parseToVec(line):
	ret = list(map(float,line.split(',')))
	print(ret)
	assert len(ret)==dim
	return ret


def get_samples_from_stdin():
	samples = []

	i=0
	while True:
		try:
			samples.append(parseToVec(input()))
		except EOFError:
			break
		i+=1 # TODO: remove this counter
	assert i==num_samples
	return samples

def squared_distance(vec1,vec2):
	assert len(vec1)==len(vec2)==dim
	return sum((vec1[i]-vec2[i])**2 for i in range(dim))

def closest_centriod_ind(sample,centriods):
	return min(range(num_clusters),key=lambda i:squared_distance(sample,centriods[i]))

def vecs_sum_iter(vecs):
	assert all(len(vec)==dim for vec in vecs)
	return map(sum,zip(*vecs))

def vec_div_iter(vec, divisor):
	return map(lambda x:x/divisor,vec)

def vecs_mean_iter(vecs):
	return vec_div_iter(vecs_sum_iter(vecs),len(vecs))

def vecs_mean(vecs):
	return list(vecs_mean_iter(vecs))

def formatVec(vec):
	return ",".join(map(str,vec))

def formatVecList(lst):
	return "\n".join(map(formatVec,lst))


all_samples = get_samples_from_stdin()
centriods = all_samples[:num_clusters]


clusters=None
for _ in range(max_iter):
	old_clusters=clusters
	clusters=[[]]*num_clusters
	for sample in all_samples:
		closest_cluster=clusters[closest_centriod_ind(sample,centriods)]
		closest_cluster.append(sample)

	# There is a one-to-one correspondence between the clusters and the centriods
	# (the clusters are a function of the centriods and the constant samples, and
	# the centriods are a function of the clusters). Thus we can check
	# for change in the clusters instead of in the centriods.
	if clusters==old_clusters: # Deep comparison
		break
	for i,cluster in enumerate(clusters):
		assert len(cluster)>0
		centriods[i]=vecs_mean(cluster)

print(formatVecList(centriods))
