import numpy as np
import kmeans_numpy
EPSILON = 0.0001
DEBUG = False

import debug_utils
dbg = debug_utils.debug_printer(False)

import numpy_utils

# TODO: use the actual soft_assert
def soft_assert(cond,msg):
    assert cond,msg

def weight_adj_mat(samples):
    # TODO: optimize this by avoiding calculation of symmetric elements
    # twice and avoiding calculation for points of the same index
    # (which we know are zeros on the diagonal).

    dists = numpy_utils.euc_dist_mat(samples, samples)
    res = np.exp(-dists / 2)
    np.fill_diagonal(res, 0)

    return res


# def weight_row_sums(samples):
#     return np.sum(weight_adj_mat(samples),axis=0)

# def diag_deg_mat(samples):
#     weight_row_sums = np.sum(weight_adj_mat(samples),axis=0)
#     return np.diag(weight_row_sums)



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
    rsqrt_row_sums = numpy_utils.row_sums(weights) ** (-0.5)

    # This is the calculation of D**(-0.5)@W@D (@ is matrix multplication) with D being the diagonal degree matfrom step 3.3.
    # In order to reduce space and performance costs, we can avoid calculating and storing the actual diagonal
    # matrix D with useless zeroes. We do that by multiplying the columns of W by D's diagonal and then
    # multiplying the result's rows by the diagonal. Since D is a diagonal matrix this is exactly the same
    # as performing the matrix multiplication.
    diag_deg_and_weight_prod = (weights * rsqrt_row_sums) * rsqrt_row_sums[:, np.newaxis]

    # TODO: consider optimizing by swapping identity matrix calculation with fill_diagonal(1-diagonal)
    return np.identity(len(weights)) - diag_deg_and_weight_prod


def qr_decomposition(mat):
    """
    Returns an orthogonal matrix q and an upper triangular matrix r
    such that r@q==mat.
    """
    dbg = debug_utils.debug_printer(False)
    dbg2 = debug_utils.debug_printer(False)
    if dbg2.is_active() or True:
        mat_copy = mat.copy()
    dim = len(mat)
    if dbg2.is_active() or True:
        expected_q,expected_r = np.linalg.qr(mat)

    # TODO: assert that the given argument mat is always symmetric
    # in our code, and optimize by not transposing it at all, using:

    # In this implementation we transpose both u and q, this way
    # we can manipulate their rows instead of columns (which is
    # faster and more readable)
    u = mat.transpose().copy()

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
        if norm==0:
           raise RuntimeError("Encountered R[i,i]=0 in qr decomposition!")

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

    # Return the transpose of q because we were looking at the transposed 
    # version up until now.
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
        q,r = qr_decomposition(e_val_mat)
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

# If k==None, uses the eigengap heuristic
def k_smallest_eigenvalue_inds(e_vals, k=None):
    if k is not None:
        return numpy_utils.argsort_k_smallest(e_vals,k)
    e_vals_first_half_sort_inds = numpy_utils.argsort_k_smallest(e_vals, len(e_vals)//2+1)
    e_vals_sorted_first_half = e_vals[e_vals_first_half_sort_inds]
    e_gaps = np.abs(np.diff(e_vals_sorted_first_half))
    dbg.print_vars({'e_vals_sorted_first_half':e_vals_sorted_first_half
        ,'e_gaps':e_gaps})
    k = np.argmax(e_gaps)+1
    dbg.print_vars(('k',k))
    return e_vals_first_half_sort_inds[:k]

def norm_spectral_cluster(samples, k=None):
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
    inds = k_smallest_eigenvalue_inds(e_vals,k)
    u = e_vec_mat[:,inds]
    if not numpy_utils.all_rows_nonzero(u):
        raise RuntimeError("U in the spectral clustering algorithm has a zero row, can not be normalized.")
    numpy_utils.normalize_rows(u)
    return kmeans_numpy.k_means(u,u.shape[1],max_iter=300)
