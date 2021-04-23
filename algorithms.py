import numpy as np


def diff_mat(a, b):
    return a[:, np.newaxis] - b


# 3.1
#
def euc_dist_mat(a, b):
    return np.linalg.norm(diff_mat(a, b), axis=-1)


#
def weight_adj_mat(samples):
    # TODO: optimize this by avoiding calculation of symmetric elements
    # twice and avoiding calculation for points of the same index
    # (which we know are zeros on the diagonal).

    dists = euc_dist_mat(samples, samples)
    res = np.exp(-dists / 2)
    np.fill_diagonal(res, 0)

    return res


# def weight_row_sums(samples):
# 	return np.sum(weight_adj_mat(samples),axis=0)

# def diag_deg_mat(samples):
# 	weight_row_sums = np.sum(weight_adj_mat(samples),axis=0)
# 	return np.diag(weight_row_sums)

# def norm_graph_lap_old(samples):
# 	return


def row_sums(mat):
    return np.sum(mat, axis=0)


def rsqrt_diag_deg_mat(samples):
    return np.diag(weight_row_sums(samples) ** (-0.5))


# 3.2, 3.3
def norm_graph_lap(samples):
    weights = weight_adj_mat(samples)

    # In these comments we will denote weighted adjacency matrix as W
    # (as in step 3.1 in the algorithm) and the
    # the diagonal degree matrix as D
    # (as in step 3.2)

    # This is the diagonal of D as shown in step 3.1
    rsqrt_row_sums = row_sums(weights) ** (-0.5)

    # This is the calculation of D**(-0.5)@W@D (@ is matrix multplication) with D being the diagonal degree matfrom step 3.3.
    # In order to reduce space and performance costs, we can avoid calculating and storing the actual diagonal
    # matrix D with useless zeroes. We do that by multiplying the columns of W by D's diagonal and then
    # multiplying the result's rows by the diagonal. Since D is a diagonal matrix this is exactly the same
    # as performing the matrix multiplication.
    diag_deg_and_weight_prod = weights * rsqrt_row_sums * rsqrt_row_sums[:, np.newaxis]

    # TODO: consider swapping identity matrix calculation with fill_diagonal(1-diagonal)
    return np.identity(len(weights)) - diag_deg_and_weight_prod


# Mutates the parameter mat!
def qr_decomposition_destructive(mat):
    u = mat.transpose()
    r = np.empty(u.shape)
    q = np.empty(u.shape)
    for i in range(len(u)):
        norm = np.linalg.norm(u[i])
        r[i][i] = norm
        normalized = q[i] = u[i] / norm

        # for j in range(i+1,len(u)):

        # 	prod = np.inner(q[i],u[j])
        # 	r[i][j] = prod
        # 	u[j] -= prod*q[i]
        prods = r[i][i + 1 :] = np.inner(normalized, u[i + 1 :])
        print(f"prods={prods}, i={i}, r={r}")
        u[i + 1 :] -= prods * q[i]

    return q.transpose(), r

