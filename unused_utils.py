
def soft_assert_nonzero_norm(cond):
    soft_assert(cond,"Encountered norm of zero while normalizing.")
    
def assert_sqmat(mat):
    return mat.ndim==2 and mat.shape[0]==mat.shape[1]
