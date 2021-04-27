#!/usr/bin/env python3

"""
Module for general matrix related algorithms, containing qr_decomposition
and qr_iteration algorithms.
"""
import numpy as np

import debug_utils
dbg = debug_utils.debug_printer(False)

EPSILON = 0.0001
def qr_decomposition(mat, out = (None, None) ,destructive=False, assume_out_r_is_up_tri=False):
    """
    Returns an orthogonal matrix q and an upper triangular matrix r
    such that r@q==mat.

    Contains an optional out parameter for specifying arrays to write
    the result into instead of allocating new ones.

    The parameter destructive, when true, allows the algorithm to
    manipulate the input array instead of copying it for its purposes.

    The parameter assume_out_r_is_up_tri is for optimization and should
    be set to true only if the caller can guarantee that the second out
    argument given is already upper triangular. Setting it to true lets
    the algorithm avoid zeroing the array. If set to true despite having
    a non upper triangular out parameter for r the returned r will not be
    upper triangular either and the algorithm will be incorrect.

    Whenever possible, it is recommended to supply a column-major
    (fortran-style) array when supplying the input matrix mat and the
    first out argument (q). This is because the algorithm uses
    operations which access elements in columns of q sequentially (and
    in this case column-major offers better cache locality).
    """
    out_q,out_r = out
    if out_q is None:
        # Column-major:
        out_q = np.empty(mat.shape,order='F')
    if out_r is None:
        out_r = np.empty(mat.shape)
    assert out_q.shape == out_r.shape == mat.shape
    if not assume_out_r_is_up_tri:
        out_r.fill(0)

    # TODO: assert that the given argument mat is always symmetric
    # in our code, and optimize by not transposing it at all, using:


    # For syntactic perference, we transpose u and q and use their
    # rows instead of columns (for readability/writeability).
    q = out_q.T
    r = out_r
    u = mat.T if destructive else mat.T.copy()

    dbg = debug_utils.debug_printer(False)
    dbg2 = debug_utils.debug_printer(False)
    if dbg2.is_active() or True:
        mat_copy = mat.copy()
    dim = len(mat)
    if dbg2.is_active() or True:
        expected_q,expected_r = np.linalg.qr(mat)


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
           raise RuntimeError("Encountered R[i,i]=0 in qr decomposition.")

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
    dbg2.print_multiline_vars({'expected_q':expected_q, 'q':out_q,
     'q.T@q':q@out_q, 'q@r':out_q@r, 'original mat':mat_copy, 'r':r,
      'expected_r':expected_r, 'expected_q,q diff':np.abs(expected_q)-np.abs(out_q)})
    return out_q, out_r


def qr_iteration(mat, destructive=False, epsilon = EPSILON):
    """
    Returns an orthogonal matrix whose columns approach the eigenvectors
    of the input matrix mat and a matrix whose diagonal elements approach
    the eigenvalues of mat.

    The parameter destructive, when true, allows the algorithm to
    manipulate the input array instead of copying it for its purposes.
    """

    #TODO: optimization - consider using constant buffers instead
    # of allocating new arrays (for example make e_val_mat have a
    # constant place in memory).

    dbg = debug_utils.debug_printer(False)

    dim = len(mat)
    e_val_mat = mat if destructive else mat.copy()
    e_vec_mat = np.identity(dim)
    q = np.empty(mat.shape)
    r = np.zeros_like(q)
    temp_mat_prod_b = np.empty(mat.shape)
    for i in range(dim):
        dbg.print_multiline_vars({
            'qr_iteration iteration number':i,
            'e_val_mat':e_val_mat,
            'e_vec_mat':e_vec_mat,
            })
        qr_decomposition(
            e_val_mat,
            out = (q,r),
            destructive=True,
            assume_out_r_is_up_tri=True
        )
        np.matmul(r,q,
                  out=e_val_mat)
        np.matmul(e_vec_mat, q,
                  out=temp_mat_prod_b)

        # Checking if we're close enough to convergence.
        # The parameter rtol is for relative tolerance, setting that to zero
        # makes the comparison non relative. atol=epsilon is our absoloute tolerance as wanted.
        is_close_to_convergence = np.allclose(
            abs(e_vec_mat), abs(temp_mat_prod_b),
            atol=epsilon, rtol=0)
        dbg.print_multiline_vars({
            'e_val_mat (after change)':e_val_mat,
            'temp_mat_prod_b':temp_mat_prod_b,
            'less than epislon cond':is_close_to_convergence
            })

        if is_close_to_convergence:
            return e_val_mat,e_vec_mat

        # Set e_vec_mat = temp_mat_prod_b and swap the buffer
        # so temp_mat_prod_b will be free to use withotu affecting
        # e_vec_mat.
        e_vec_mat, temp_mat_prod_b = temp_mat_prod_b , e_vec_mat
    return e_val_mat,e_vec_mat
