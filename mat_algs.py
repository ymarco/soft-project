#!/usr/bin/env python3

"""
Module for general matrix related algorithms, containing qr_decomposition
and qr_iteration algorithms.
"""

import numpy as np

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
    # For syntactic perference, we transpose u and q and use their
    # rows instead of columns (for readability/writeability).
    q = out_q.T
    r = out_r
    u = mat.T if destructive else mat.T.copy()

    dim = len(mat)

    for i in range(dim):
        norm = np.linalg.norm(u[i])
        r[i,i] = norm



        # As instructed on the forum, we exit on R[i,i]=0
        if norm==0:
           raise ZeroDivisionError(
               "Encountered R[i,i]=0 in qr decomposition.")

        normalized = q[i] = u[i] / norm

        ## Alternative option - when r[i][i]==0 assign q[i]=0.
        #if norm==0:
            #normalized = q[i] = 0
        #else:
            #normalized = q[i] = u[i] / norm


        # The following code is equivalent to:
            # for j in range(i+1,len(u)):
            #     prod = np.inner(q[i],u[j])
            #     r[i,j] = prod
            #     u[j] -= prod*q[i]

        remaining_vecs = u[i+1 :]
        prods = r[i,i + 1 :] = np.inner(normalized, remaining_vecs)
        remaining_vecs -= prods[:, np.newaxis] * q[i]

    return out_q, out_r


def qr_iteration(mat, destructive=False, epsilon = EPSILON):
    """
    Returns an orthogonal matrix whose columns approach the eigenvectors
    of the input matrix mat and a matrix whose diagonal elements approach
    the eigenvalues of mat.

    The parameter destructive, when true, allows the algorithm to
    manipulate the input array instead of copying it for its purposes.

    For performance reasons, whenever using destructive=True, it is
    recommended to supply a column-major (fortran-style) array when
    supplying the input matrix mat. (Because mat is used in qr_iteration,
    see the documentation there). This performance gain is backed up
    by measurements.
    """

    dim = len(mat)

    # Use column-major copy for better cache locality, see documentation
    # above and in qr_decomposition.
    e_val_mat = mat if destructive else mat.copy(order='F')

    e_vec_mat = np.identity(dim)

    # Again, column-major for cache locality (see qr_decompisition doc).
    # Another possible reason for why this yields better performence
    # (according to benchmarks) except for the one documented in
    # qr_decompisition is that in the matrix multiplication performed
    # in this algorithm q the right multiplicand (sums are across its'
    # columns).
    q = np.empty(mat.shape, order='F')

    r = np.zeros_like(q)
    temp_mat_prod_b = np.empty(mat.shape)
    for i in range(dim):
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
        # makes the comparison non relative. atol=epsilon is our absolute tolerance as wanted.
        is_close_to_convergence = np.allclose(
            abs(e_vec_mat), abs(temp_mat_prod_b),
            atol=epsilon, rtol=0)

        if is_close_to_convergence:
            return e_val_mat,e_vec_mat

        # Set e_vec_mat = temp_mat_prod_b and swap the buffer
        # so temp_mat_prod_b will be free to use withotu affecting
        # e_vec_mat.
        e_vec_mat, temp_mat_prod_b = temp_mat_prod_b , e_vec_mat
    return e_val_mat,e_vec_mat
