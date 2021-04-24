import numpy as np
EPSILON = 0.0001
DEBUG = False

import debug_utils
dbg = debug_utils.debug_printer(False)

# TODO: use the actual soft_assert
def soft_assert(cond,msg):
    assert cond,msg

def soft_assert_nonzero_norm(cond):
    soft_assert(cond,"Encountered norm of zero while normalizing.")

def assert_sqmat(mat):
    return mat.ndim==2 and mat.shape[0]==mat.shape[1]

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
#     return np.sum(weight_adj_mat(samples),axis=0)

# def diag_deg_mat(samples):
#     weight_row_sums = np.sum(weight_adj_mat(samples),axis=0)
#     return np.diag(weight_row_sums)


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
    diag_deg_and_weight_prod = (weights * rsqrt_row_sums) * rsqrt_row_sums[:, np.newaxis]

    # TODO: consider optimizing by swapping identity matrix calculation with fill_diagonal(1-diagonal)
    return np.identity(len(weights)) - diag_deg_and_weight_prod


# Mutates the parameter mat!
def qr_decomposition_destructive(mat):
    dbg = debug_utils.debug_printer(False)
    dbg2 = debug_utils.debug_printer(True)
    if dbg2.is_active():
        mat_copy = mat.copy()
    dim = len(mat)
    if dbg2.is_active():
        expected_q,expected_r = np.linalg.qr(mat)

    u = mat.transpose()
    q = np.zeros(u.shape)
    r = np.zeros(u.shape)
    dbg.print_multiline_vars({'u':u,'r':r,'q':q})
    dbg2.print_multiline_vars({'u':u,'r':r,'q':q})
    for i in range(dim):
        norm = np.linalg.norm(u[i])
        r[i,i] = norm

        if i==dim-1:
            dbg.set_active(True)
        dbg.print_vars({'i':i})
        dbg.print_vars({'norm':norm})

        # Exit on r[i,i]==0 as instructed on the forum.
        soft_assert(norm!=0, "Encountered R[i,i]=0 in qr decomposition!")
        dbg.print_multiline_vars({'q[i]':q[i], 'u[i]':u[i], 'norm':norm})
        normalized = q[i] = u[i] / norm #if norm != 0 else np.zeros(dim)
        dbg.print_vars({'normalized':normalized})
        dbg.print_multiline_vars({'u':u,'r':r,'q':q})

        # for j in range(i+1,len(u)):

        #     prod = np.inner(q[i],u[j])
        #     r[i,j] = prod
        #     u[j] -= prod*q[i]
        
        remaining_vecs = u[i+1 :]
        if dbg.is_active():
            prods_calc = np.inner(normalized, remaining_vecs)
            dbg.print_multiline_vars({'prods-calc':prods_calc})
        prods = r[i,i + 1 :] = np.inner(normalized, remaining_vecs)
        dbg.print_multiline_vars({'prods':prods, 'r':r})
        dbg2.print_multiline_vars({'prods[:, np.newaxis] * q[i]':prods[:, np.newaxis] * q[i]})
        remaining_vecs -= prods[:, np.newaxis] * q[i]

    dbg2.print("calculated q & r:")
    dbg2.print_multiline_vars({'expected_q':expected_q, 'q':q.T,
     'q.T@q':q@q.T, 'q@r':q.T@r, 'original mat':mat_copy, 'r':r, 'expected_r':expected_r})
    return q.transpose(), r


def qr_iteration(mat):
    dim = len(mat)
    e_val_mat = mat
    e_vec_mat = np.identity(dim)
    for _ in range(dim):
       q,r = qr_decomposition_destructive(e_val_mat)
       e_val_mat = r@q
       mat_prod = e_vec_mat@q
       if np.all(np.abs(np.abs(q)-np.abs(mat_prod))<=EPSILON):
           return e_val_mat,e_vec_mat
       e_vec_mat = mat_prod
    return e_val_mat,e_vec_mat


def row_norms(mat):
    return np.linalg.norm(mat,axis=-1)

def normalize_rows(mat):
   mat /= row_norms(mat)[:,np.newaxis]

def all_rows_nonzero(mat):
    np.all(np.any(mat!=0,axis=-1))

def norm_spectral_cluster(samples):
    dim = len(samples)
    l = norm_graph_lap(samples)

    # TODO: rename e_val_mat to e_val_c_mat and e_vec_mat to e_vec_d_mat
    # (for columns and diagonal, respectively)
    e_val_mat,e_vec_mat = qr_iteration(l)
    e_vals = e_val_mat.diagonal()

    # TODO: consider optimizing with np.argpartition(e_vals, np.arange(dim/2)) instead of full sort
    e_vals_first_half_sort_inds = np.argsort(e_vals)[:dim//2+1]
    e_vals_sorted_first_half = e_vals[e_vals_first_half_sort_inds]
    e_gaps = np.abs(np.diff(e_vals_sorted_first_half))
    k = np.argmax(e_gaps)
    relevant_e_vec_inds = e_vals_first_half_sort_inds[:k]
    u = e_vec_mat[:,relevant_e_vec_inds]

    soft_assert(all_rows_nonzero(u), "U in the spectral clustering algorithm has a zero row, can not be normalized.")
    normalize_rows(u)
    # TODO: use kmeans on u.
    return u
