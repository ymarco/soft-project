import numpy as np

from debug_utils import *

#dbg = debug_printer(True)

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

def k_means_pp(samples, k):
	return None

def k_means(samples, k=None, initial_centroids = None, max_iter=1000):
	if initial_centroids is None and k is None:
		raise ValueError("Either k or initial_centroids must be supplied")
	elif initial_centroids is None:
		centroids = k_means_pp(samples,k)
	else:	
		centroids = initial_centroids.copy()
	num_samples = len(samples)
	assert 1<=len(centroids)<=num_samples

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
			print("lol")
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

