"""
Module to handle all regression functions.

Largely taken from Alex Huth's regression functions

"""

### --- Imports --- ###
import warnings
import numpy as np
import scipy.stats as _stats
import time
from . import vmt_io
import vmt_utils as _utils 
from .Stats import utils as _sutils

### --- Parameters --- ###    
DEFAULT_ALPHAS=np.array([0]+[2**x for x in range(10,21)])
# DEFAULT_ALPHAS = np.logspace(0,4,10)

del x

def _fit_ridge_alpha(trn_fs,trn_data,val_fs,val_data,alphas=DEFAULT_ALPHAS,
    chunk_sz=5000,is_efficient=True,dtype=np.single, is_verbose=False, pthr=0.005,
    square_alpha=False,return_resids=False):
    """Get prediction correlations for a set of alphas on val_data, without ever computing weights on trn_fs

    Uses ridge regression to find a linear transformation of `trn_fs` that approximates `trn_data`.
    Then tests by comparing the transformation of `val_fs` to `val_data`. This procedure is repeated
    for each regularization parameter (alpha) in `alphas`. The correlation between each prediction and
    each response for each alpha is returned. Note that the regression weights are NOT returned.
    
    This is more efficient than full ridge regression (with weight computation); it is meant to be 
    used inside other ridge functions (after data has been split into bootstrap / cross-validation 
    splits) to find optimal alpha values. 
    

    Parameters
    ----------
    trn_fs : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    trn_data : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    val_fs : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    val_data : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the Frobenius norm of trn_fs. Good for rigorously
        comparing models with different numbers of parameters.
    dtype : np.dtype
        All data will be cast as this dtype for computation. np.single is used by default for memory
        efficiency.
    singcutoff : float [WIP: not implemented yet]
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus trn_fs. If trn_fs is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.

    Returns
    -------
    trn_corrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of val_data for each alpha.
    
    """    
    n_tps,n_voxels = trn_data.shape
    n_chunks = np.ceil(n_voxels/np.float(chunk_sz)).astype(np.int32)
    cc = np.zeros((n_voxels,len(alphas)),dtype=dtype)
    if return_resids:
        resids = np.zeros((n_tps,n_voxels,len(alphas)),dtype=dtype)
    pred_A = []
    if is_efficient:
        # Efficient Ridge regression from A. Huth, Part (1):
        # Full multiplication for validation (here, random split of
        # training data) prediction is: 
        # pred = (Xval*Vx) * Dx * (pinv(Ux)*Ychunk)   # NOTE: pinv(Ux) = Ux'
        # We will pre-compute the first and third terms in parentheses:
        # pred =   XvalVx  * Dx *  UxYchunk
        if is_verbose: 
            print('->Doing SVD of stimulus design matrix')
            t0 = time.time()
            #time.sleep(.01); # To ensure printing?
        m,n = trn_fs.shape
        if m>n:
            Ux,Sx,Vx = _utils._svd(trn_fs,full_matrices=False)
        else:
            Vx,Sx,Ux = _utils._svd(trn_fs.T,full_matrices=False)
            # Switcheroo of Vx and Ux due to transpose of input matrix
            Ux = Ux.T
            Vx = Vx.T

        if is_verbose:
            t1 = time.time()
            print('->Done with SVD in %0.2f sec'%(t0-t1))
        # For more efficient computation:
        #k = len(Sx) 
        ## OR: 
        ## singcutoff = (XX);
        ## k = sum(sx > singcutoff);
        ## sx = sx(1:k);
        XvalVx = val_fs.dot(Vx.T) # NOTE: IN MATLAB, No Vx', because Matlab leaves V in transposed form!
    else:
        raise NotImplementedError("Sorry, not done yet!")

    for iChunk in range(n_chunks):
        print('Running chunk %d of %d...\n'%(iChunk+1,n_chunks))
        ChIdx = np.arange(chunk_sz) + chunk_sz*iChunk
        ChIdx = ChIdx[ChIdx<n_voxels] # clip extra voxels in last run.
        Ychunk = trn_data[:,ChIdx]

        # Fit model with all lambdas (for subset of voxels)
        if not is_efficient:
            raise Exception('LAME! no slow reliable ridge implemented.')
            #[Wt L] = ridgemulti(X,Ychunk,params.lambdas);
        else:
            # Efficient Ridge regression from A. Huth, part (2)
            # NOTE: weights are never explicitly computed!
            UxYchunk = Ux.T.dot(Ychunk)
        
        if is_verbose:
            print('Checking model predictions...')
        for iA,A in enumerate(alphas):
            if not is_efficient:
                pred = np.cast(np.single)[Xval.dot(Wt[:,:,iA])]
            else:
                # Efficient Ridge regression from A. Huth, part (3)
                # Normalize lambda by Frobenius norm for stim matrix
                aX = A # * norm(X,'fro'); # ... or not
                # Need to decide for final whether aX**2 or not
                if square_alpha:
                    Dx = Sx/(Sx**2 + aX**2) 
                else:
                    Dx = Sx/(Sx**2 + aX) 
                # Compute predicitons (XvalVx and UxYchunk computed above)
                # (mult diag is slightly faster than matrix multiplication in timing tests)
                pred = _utils.mult_diag(Dx, XvalVx, left=False).dot(UxYchunk) 
            # Compute prediction accuracy (correlations)
            cc[ChIdx,iA]=_sutils.column_corr(pred,val_data[:,ChIdx])
            if return_resids:
                resids[:,ChIdx,iA] = val_data[:,ChIdx]-pred
    if return_resids:
        return cc,resids
    else:
        return cc

def ridge_cv(trn_fs, trn_data, val_fs=None, val_data=None, alphas=DEFAULT_ALPHAS, 
    n_resamps=10, n_splits=10, chunk_sz=5000, dtype=np.single,pthr=0.005,
    square_alpha=False,is_verbose=False):
    """Compute ridge regression solution for beta weights and predictions of validation data.
    
    Regularization parameter (alpha) is computed by cross-validation within the 
    training data (trn_fs and trn_data). 
    
    Validation predictions and correlations are returned if val_fs and val_data are provided.
    

    Parameters
    ----------
    pthr : float in [0..0.05]
        Used to select the alpha parameter. For each alpha tested, pthr is used to define a minimum 
        significant correlation (r_sig). The function then computes the number voxels with training-set 
        correlations greater than r_sig minus the number of responses with correlation less than -r_sig
        This is a vague metric of non-centered skewness, and works better (?) than the mean correlation
        across voxels to select an optimal alpha parameter.

    Uses ridge regression with a bootstrapped held-out set to get a single optimal alpha values for all voxels.
    [n_chunks] random chunks of length [chunklen] will be taken from [trn_fs] and [trn_data] for each regression
    run. [nboots] total regression runs will be performed.
    """
    #def ridge_cv(model,data,n_splits=10,n_resamps=10,alpha=DEFAULT_ALPHAS,efficient=np.nan,):
    #   (trn_fs, trn_data, val_fs, val_data, alphas, nboots, chunklen, n_chunks, dtype=np.single, corrmin=0.2):
    n_resp, n_voxels = trn_data.shape
    _,n_channels = trn_fs.shape
    n_chunks = np.ceil(n_voxels/np.float(chunk_sz)).astype(np.int32)
    bestalphas = np.zeros((n_resamps, n_voxels))  ## Will hold the best alphas for each voxel
    trn_idx,val_idx = _utils.contig_partition(trn_fs.shape[0],n_splits)

    Rcmats = np.zeros((n_voxels,len(alphas),n_resamps))
    for iRpt,cvi in enumerate(np.random.permutation(n_splits)[:n_resamps]):
        if is_verbose:
            print('Running split %d/%d'%(iRpt+1,n_resamps))
        ti,vi = trn_idx[cvi],val_idx[cvi]
        trn_fs_split = trn_fs[ti,:]
        val_fs_split = trn_fs[vi,:]
        trn_data_split = trn_data[ti,:]
        val_data_split = trn_data[vi,:]
        
        # Run ridge regression to estimate predictions (within training set) for different alphas
        Rcmats[:,:,iRpt] = _fit_ridge_alpha(trn_fs_split, trn_data_split, val_fs_split, val_data_split, alphas,
                             dtype=dtype, chunk_sz=chunk_sz,pthr=pthr,square_alpha=square_alpha)
    if is_verbose:
        print("Finding best alpha...")
    ## Find best alpha for each voxel
    #trncc_byvox = np.nansum(Rcmats,axis=2)/np.sum(np.logical_not(np.isnan(Rcmats)),axis=2)
    trncc_byvox = np.nanmean(Rcmats,axis=2)
    # Taking mean is BS: too many voxels poorly predicted, floor effect. 
    #mean_cv_corr = np.nanmean(mean_cv_corr_byvox,axis=0)
    #bestalphaind = np.argmax(mean_cv_corr)
    # Thus just count voxels over significance threshold (with a lenient threshold)
    #print(len(vi))
    sig_thresh = _sutils.pval2r(pthr,len(vi),is_two_sided=False)
    n_sig_vox_byalpha = sum(trncc_byvox>sig_thresh)-sum(trncc_byvox<-sig_thresh)
    bestalphaind = np.argmax(n_sig_vox_byalpha)
    alpha = alphas[bestalphaind]
    if is_verbose:
        print("Best alpha = %0.3f"%alpha)
    
    ## Find weights for each voxel
    U,S,Vt = np.linalg.svd(trn_fs, full_matrices=False)
    # Loop over groups of voxels
    wt = np.zeros((n_channels,n_voxels),dtype=dtype)

    ###
    if (not val_fs is None) and (not val_data is None):
        # Validation data / model supplied implies we want predictions
        do_pred = True
        if np.sum(val_fs[:,0]) != val_fs.shape[0]:
            warnings.warn('First column of val_fs is NOT all ones! Consider including a DC term!')
        # Pre-allocate for predictions, with or without separate validation sequences to predict
        is_rpts = np.ndim(val_data)==3
        if is_rpts:
            n_rpts,n_tps_val,n_voxels_val = val_data.shape
            cc = np.zeros((n_rpts,n_voxels),dtype);
        else:
            n_rpts,(n_tps_val,n_voxels_val) = 0,val_data.shape
            cc = np.zeros((n_voxels),dtype);
        pred = np.zeros((n_tps_val,n_voxels_val),dtype)
    else:
        # No Validation data / model supplied
        do_pred = False;

    if is_verbose:
        predstr = ' and model predictions...' if do_pred else "..."
        print('Computing weights'+predstr)
    for iChunk in range(n_chunks):
        if is_verbose:
            print('Running chunk %d of %d...\n'%(iChunk+1,n_chunks))
        ChIdx = np.arange(chunk_sz) + chunk_sz*iChunk
        ChIdx = ChIdx[ChIdx<n_voxels] # clip extra voxels in last run.
        Ychunk = trn_data[:,ChIdx]

        UtYchunk = np.dot(U.T, np.nan_to_num(Ychunk))
        if square_alpha:
            wt[:,ChIdx] = reduce(np.dot, [Vt.T, np.diag(S/(S**2+alpha**2)), UtYchunk])
        else:
            wt[:,ChIdx] = reduce(np.dot, [Vt.T, np.diag(S/(S**2+alpha)), UtYchunk])

        ## Find test correlations if validation data is present
        if do_pred:
            # Compute correlations btw validation data and model prediction
            pred[:,ChIdx] = val_fs.dot(wt[:,ChIdx]).astype(dtype) 
            nnpred = np.nan_to_num(pred[:,ChIdx])
            if is_rpts:
                # The transpose here is related to deep mysteries in python. See 
                cc[:,ChIdx] = np.vstack([_sutils.column_corr(nnpred,val_data[rpt,:,ChIdx].T) for rpt in range(n_rpts)])
            else:
                cc[ChIdx] = _sutils.column_corr(nnpred,val_data[:,ChIdx])

    # Output
    out = dict(
        weights=wt,
        alpha=alpha,
        n_sig_vox_byalpha=n_sig_vox_byalpha,
        #trncc_byvox=trncc_byvox,f
        #trncc_byvox_byalpha=Rcmats
        )
    if not val_data is None:    
        out['cc'] = cc
    return out

