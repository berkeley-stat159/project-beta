# General utility functions for vm_tools
"""
These functions are meant to manipulate and reshape data, to facilitate bootstrapping, 
Statistical functions (those concerned with prediction accuracy, significance levels)
"""
import numpy as np

def _svd(M,**kwargs):
    """
    same as np.linagle.svd() function
    add exception handle
    """
    try:
        U,S,Vh = np.linalg.svd(M,**kwargs)
    except np.linalg.LinAlgError as e:
        print "NORMAL SVD FAILED, trying more robust dgesvd.."
        from .svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(M,**kwargs)
    return U,S,Vh

def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T 
    else:
        return d*mtx


# def split_data(n_splits,n_tps,is_contiguous=True):
#   """Splits data of length "n_tps" up in to "n_splits" cross-validation chunks
    
#   Parameters
#   ----------
#   n_splits : scalar
#       number of splits
#   n_tps : scalar
#       number of time points to subdivide
#   is_contiguous : bool
#       Governs whether chunks are contiguous
    
#   Returns a list ("splits") of (trnIdx,valIdx) pairs of indices
#   (Each split is n_tps/n_splits long)
#   """
#   splits = []
#   n_per_split = n_tps/n_splits
#   if is_contiguous:
#       indices = np.arange(n_tps)
#   else:
#       indices = np.random.permutation(n_tps)
#   for ii in range(n_splits):
#       vIdx = indices[ii*n_per_split:(ii+1)*n_per_split]
#       t1 = indices[0:ii*n_per_split]
#       t2 = indices[(ii+1)*n_per_split:]
#       tIdx = np.concatenate((t1,t2))
#       splits.append((tIdx,vIdx))
#   return splits

def add_lags(X,lags=(2,3,4)):
    """
    Add lags to design matrix (X)
    X should be a TIME x CHANNELS matrix
    lags should be a list of time points (nTRs from stim onset), e.g. [2,3,4]
    """
    nlags = len(lags)
    lags = [x-1 for x in lags] # make lags 0-first python indices
    nTP,nChan = X.shape
    XA = np.zeros((nTP,nChan*nlags))
    for iL,L in enumerate(lags):
        XA[:,nChan*iL:nChan*(iL+1)] = np.concatenate((np.zeros((L,nChan)),X[:nTP-L,:]),axis=0)
    X = XA
    return X

def add_constant(X,is_first=True):
    """
    Add a constant to a design matrix
    X should be a TIME x CHANNELS matrix
    is_first default is True (= add column of ones on LEFT); False = Right
    """
    if is_first:
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
    else:
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
    return X


def contig_partition(n_datapts,n_splits):
    """Evenly partition n_datapts into n_splits

    Each pair is a partition of X, where validation is an iterable
    of length n_datapts/n_splits. 
    """
    nn = np.cast['int32'](np.ceil(np.linspace(0,n_datapts,n_splits+1)))
    val = [np.arange(st,fn) for st,fn in zip(nn[:-1],nn[1:])]
    trn = [np.array([x for x in range(n_datapts) if not x in v]) for v in val]
    return trn,val
