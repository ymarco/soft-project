import numpy as np
import kmeans_numpy
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
    dbg2 = debug_utils.debug_printer(False)
    if dbg2.is_active() or True:
        mat_copy = mat.copy()
    dim = len(mat)
    if dbg2.is_active() or True:
        expected_q,expected_r = np.linalg.qr(mat)

    # TODO: assert that the given argument mat is always symmetric
    # in our code, and optimize by not transposing it at all, using:
    # u = mat
    u = mat.transpose()

    q = np.empty(u.shape)
    r = np.zeros(u.shape)
    dbg.print_multiline_vars({'u':u,'r':r,'q':q})
    dbg2.print_multiline_vars({'u':u,'r':r,'q':q})
    for i in range(dim):
        norm = np.linalg.norm(u[i])
        r[i,i] = norm

        DEBUG_LAST_QR_DECOMP_ITER = False
        if i==dim-1 and DEBUG_LAST_QR_DECOMP_ITER:
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

    q=q.transpose()
    dbg2.print("calculated q & r:")
    dbg2.print_multiline_vars({'expected_q':expected_q, 'q':q,
     'q.T@q':q.T@q, 'q@r':q@r, 'original mat':mat_copy, 'r':r,
      'expected_r':expected_r, 'expected_q,q diff':np.abs(expected_q)-np.abs(q)})
    return q, r


def qr_iteration(mat):

    #TODO: optimization - consider using constant buffers instead
    # of allocating new arrays (for example make e_val_mat have a
    # constant place in memory).

    dbg = debug_utils.debug_printer(False)
    dim = len(mat)
    e_val_mat = mat
    e_vec_mat = np.identity(dim)
    for i in range(dim):
        dbg.print_multiline_vars({
            'qr_iteration iteration number':i,
            'e_val_mat':e_val_mat,
            'e_vec_mat':e_vec_mat,
            })
        q,r = qr_decomposition_destructive(e_val_mat)
        e_val_mat = r@q
        mat_prod = e_vec_mat@q

        is_close_to_convergence = np.allclose(abs(e_vec_mat), abs(mat_prod), atol=EPSILON, rtol=0)
        dbg.print_multiline_vars({
            'e_val_mat (after change)':e_val_mat,
            'mat_prod':mat_prod,
            'less than epislon cond':is_close_to_convergence
            })

        # Checking if we're close enough to convergence.
        # The parameter rtol is for relative tolerance, setting that to zero
        # makes the comparison non relative. atol=EPSILON is our absoloute tolerance as wanted. 
        if is_close_to_convergence:
            return e_val_mat,e_vec_mat
        e_vec_mat = mat_prod
    return e_val_mat,e_vec_mat


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
# If k==None, uses the eigengap heuristic
def k_smallest_eigenvalue_inds(e_vals, k=None):
    if k is not None:
        return argsort_k_smallest(e_vals,k)
    e_vals_first_half_sort_inds = argsort_k_smallest(e_vals, len(e_vals)//2+1)
    e_vals_sorted_first_half = e_vals[e_vals_first_half_sort_inds]
    e_gaps = np.abs(np.diff(e_vals_sorted_first_half))
    dbg.print_vars({'e_vals_sorted_first_half':e_vals_sorted_first_half
        ,'e_gaps':e_gaps})
    k = np.argmax(e_gaps)+1
    dbg.print_vars(('k',k))
    return e_vals_first_half_sort_inds[:k]

def norm_spectral_cluster(samples):
    dbg = debug_utils.debug_printer(False)
    dim = len(samples)
    l = norm_graph_lap(samples)

    # TODO: rename e_val_mat to e_val_c_mat and e_vec_mat to e_vec_d_mat
    # (for columns and diagonal, respectively)

    # TODO: check if adding a bit to the diagonal yeilds better clusters
    PRE_QR_ITERATION_ADDEND = 0 if True else (2*EPSILON)*np.identity(len(l))
    l+=PRE_QR_ITERATION_ADDEND
    e_val_mat,e_vec_mat = qr_iteration(l)
    e_vals = e_val_mat.diagonal()
    dbg.d_assert(np.all(e_vals>=-EPSILON),
        "Encountered negative eigenvalues, "
        + debug_utils.vars_to_str(('e_vals',e_vals)))
    dbg.print_vars({'e_val_mat':e_val_mat, 'e_vals':e_vals, 'e_vec_mat':e_vec_mat})
    inds = k_smallest_eigenvalue_inds(e_vals)
    u = e_vec_mat[:,inds]
    dbg.d_assert(all_rows_nonzero(u), '\n'+debug_utils.vars_to_multiline_str(('U',u)))
    soft_assert(all_rows_nonzero(u), "U in the spectral clustering algorithm has a zero row, can not be normalized.")
    normalize_rows(u)
    # TODO: use kmeans on u.
    return kmeans_numpy.k_means(u,u.shape[1])
