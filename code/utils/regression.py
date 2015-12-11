"""
Module to handle all regression functions.

Largely taken from Alex Huth's regression functions

Notes:
Split off predictions from fitting? 
Advantages: cleaner code
Disadvantages: less efficient - need to run through chunking code 2x

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

def compute_vif(X,**kwargs):
    """Compute Variance Inflation Factor for a design matrix
    
    Parameters
    ----------
    X : array-like
        design matrix of variables; time x channels
    kwargs : dict
        named inputs to vmt.regression.ols
        
    Returns
    -------
    VIF : array-like, 1D
        vector of Variance Inflation Factors (one for each channel [column] in X)
    """
    n_tps,n_chans = X.shape
    if n_chans>n_tps:
        raise ValueError("Number of channels cannot be greater than number of time points\n for Variance Inflation Factor computation!")
    VIF = np.zeros((n_chans,))
    for ic,c in enumerate(X.T):
        if len(X.T)>200:
            if ic%200==0:
                print('Computing VIF for channel %d'%ic)
        ci = np.arange(n_chans)!=ic
        out = ols(X[:,ci],X[:,ic][:,None])
        y_hat = (X[:,ci].dot(out['weights'])).flatten()
        R2 = 1-(np.var(X[:,ic]-y_hat)/np.var(X[:,ic]))
        VIF[ic] = 1/(1-R2)
    return VIF

def ols(trn_fs,trn_data,val_fs=None,val_data=None,chunk_sz=5000,dtype=np.single,is_verbose=False,input_weights=None):
    """
    Parameters
    ----------
    trn_fs : array-like
        feature space array, time x channels; representation of a stimulus
    val_fs : array-like
        feature space array, time x channels; representation of a stimulus
    trn_data : array-like
        data array, time x voxels
    val_data : array-like
        data array, time x voxels
    chunk_sz : scalar
        maximum number of voxels to analyze at once (constrains matrix multiplication size for large data sets)

    Other Parameters
    ----------======
    is_verbose : bool
        verbose / not
    dtype : np.dtype
        data type for weights / predictions
    input_weights : array-like, 1D
        For weighted least squares... Frankly IDKWTF this is useful for. Look it up.
        Default (None) creates identity matrix (i.e. has no effect) (See code)
    """

    # Check on first column    
    if np.sum(trn_fs[:,0])!=trn_fs.shape[0]:
        warnings.warn('First column of trn_fs is NOT all ones! Consider including a DC term!')
    # Size of matrices
    n_tps,n_voxels = trn_data.shape
    _,n_channels = trn_fs.shape

    # For weighted least squares, if desired 
    if input_weights is None:
        W = np.eye(n_tps)
    else:
        W = np.diag(1/input_weights**2)
    # Compute pseudo-inverse of (weighted) squared design matrix
    XtXinv = np.linalg.pinv(trn_fs.T.dot(W.dot(trn_fs))) 

    if (not val_fs is None) and (not val_data is None):
        # Validation data / model supplied implies we want predictions
        Xv = val_fs
        do_pred = True
        if np.sum(Xv[:,0]) != Xv.shape[0]:
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
    # Pre-allocate variables
    weights = np.zeros((n_channels,n_voxels),dtype=dtype)
    # Divide data into chunks if necessary for memory saving:
    n_chunks = np.ceil(n_voxels/float(chunk_sz)).astype(np.uint32)
    for iChunk in range(n_chunks):
        if is_verbose and (n_chunks>1): 
            print('Running chunk %d of %d...'%(iChunk+1,n_chunks))
        ChIdx = np.arange(chunk_sz) + chunk_sz*iChunk
        ChIdx = ChIdx[ChIdx<n_voxels] # clip extra voxels in last run.
        Ychunk = trn_data[:,ChIdx]
        # 'reduce' takes the dot products of the matrices in order from left to right
        weights[:,ChIdx] = reduce(np.dot,[XtXinv,trn_fs.T,W,Ychunk])
        
        if do_pred:
            if is_verbose: 
                print('Obtaining model predictions...')             
            # Compute correlations btw validation data and model prediction
            pred[:,ChIdx] = Xv.dot(weights[:,ChIdx]).astype(dtype) 
            if is_rpts:
                # The transpose here is related to deep mysteries in python. See 
                cc[:,ChIdx] = np.vstack([_sutils.column_corr(pred[:,ChIdx],val_data[rpt,:,ChIdx].T) for rpt in range(n_rpts)])
            else:
                cc[ChIdx] = _sutils.column_corr(pred[:,ChIdx],val_data[:,ChIdx])
    out = dict(weights=weights)
    if do_pred:
        out.update(dict(pred=pred,cc=cc))
    return out  

def ridge(trn_fs, trn_data, val_fs=None, val_data=None, alpha=0, 
    chunk_sz=5000, dtype=np.single,square_alpha=False,is_verbose=False):
    """Vanilla ridge regression.
    
    Regularization parameter (alpha) must be supplied (for computation of regularization parameter,
        see ridge_cv or ridge_boot) 
    
    Validation predictions and correlations are returned if val_fs and val_data are provided.
    

    Parameters
    ----------

    Returns
    -------
    """
    ## --- Housekeeping --- ###
    n_resp, n_voxels = trn_data.shape
    _,n_channels = trn_fs.shape
    n_chunks = np.ceil(n_voxels/np.float(chunk_sz)).astype(np.int32)

    ### --- Set up SVD-based weight computation --- ###
    U,S,Vt = np.linalg.svd(trn_fs, full_matrices=False)

    ### --- Set up predictions --- ###
    if (not val_fs is None) and (not val_data is None):
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

    ### --- Loop over groups of voxels to compute weights & predictions --- ###
    wt = np.zeros((n_channels,n_voxels),dtype=dtype)
    if is_verbose:
        predstr = ' and model predictions...' if do_pred else "..."
        print('Computing weights'+predstr)
    for iChunk in range(n_chunks):
        if is_verbose and (n_chunks>1):
            print('Running chunk %d of %d...\n'%(iChunk+1,n_chunks))
        ChIdx = np.arange(chunk_sz) + chunk_sz*iChunk
        ChIdx = ChIdx[ChIdx<n_voxels] # clip extra voxels in last run.
        Ychunk = trn_data[:,ChIdx]
        UtYchunk = np.dot(U.T, np.nan_to_num(Ychunk))
        if square_alpha:
            wt[:,ChIdx] = reduce(np.dot, [Vt.T, np.diag(S/(S**2+alpha**2)), UtYchunk])
        else:
            wt[:,ChIdx] = reduce(np.dot, [Vt.T, np.diag(S/(S**2+alpha)), UtYchunk])
        ### --- Find test correlations if validation data is present --- ###
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
        #trncc_byvox=trncc_byvox,
        #trncc_byvox_byalpha=Rcmats
        )
    if not val_data is None:    
        out['cc'] = cc
    return out

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

# def fit_joint_model_fMRI(dbi, trn_fs, trn_data, val_fs=None, val_data=None, reg='ridge_cv',
#     add_dc=False, noise_preds=None, lags=[2,3,4], chunk_sz=5000, save_weights=False,
#     dtype=np.single, run_local=False, is_overwrite=False, sdir='/auto/k8/mark/fMRIDB/',
#     pred_metrics=('cc','ccFull','valPred')):
    
#     # Needs to deal w/ different alphas, potentially with different models being fit. 
#     # Pass out parallelized model runs if the models haven't been run individually.

#     pass    
def ridge_joint(trn_fs,trn_data,alphas,val_fs=None,val_data=None,square_alphas=False,chunk_sz=5000,is_verbose=True):
    """

    Alphas are required. Should be one per model, we're not fitting them here.

    trn_fs should be a list of the matrices of the feature spaces you wish to concatenate.


    Stim should already have any lags applied to them
    """

    n_resp, n_voxels = trn_data.shape
    n_channels = np.sum([tfs.shape[1] for tfs in trn_fs])
    n_chunks = np.ceil(n_voxels/np.float(chunk_sz)).astype(np.int32)

    if square_alphas:
        alphas = [a**2 for a in alphas]

    num_train_points = float(list(trn_data.shape)[0])
    num_val_points = float(list(val_data.shape)[0])

    n_abc = [np.minimum(*t_fs.shape) for t_fs in trn_fs]
    ######################################################################################
    ### --- First up: compute modified covariance matrix & scaled/rotated stimulus --- ###
    ######################################################################################
    # Perform SVD on training sets for all three models 
    if is_verbose:
        print("computing SVD")
    U_trn,W_trn,Vt_trn = [],[],[]
    for t_fs in trn_fs:
        uu,ww,vv = np.linalg.svd(trn_stim_A, full_matrices=False)
        U_trn.append(uu)
        W_trn.append(ww)
        Vt_trn.append(vv)

    # The square of Ws (the singular values from the SVD) are the eigenvalues of the covariance matrix but have not been divided by n-1.
    L = [ww**2/float(num_train_points-1) for ww in W_trn]

    ### --- IDK WTF. Ask Wendino. --- ###
    ## to change: make sure that Ws are in the right units (divided by n-1) when bootstrapping, so that alphas are already in correct units
    ## at that point you can change the lines below and not divide alpha by (n-1)
    # TO DO: make this more than one line for clarity.
    w_alpha_trn = [np.diag(np.sqrt(1./(LL + aa)/(num_train_points-1))) for LL,aa in zip(L,alphas)]
    #w1_alpha_trn = sqrt(1./(L1+ alphas_A2[0]/(num_train_points-1))) 
    #w1_alpha_trn = diag(w1_alpha_trn) #%turn it from an array to a matrix

    # Create & combine rotated & scaled stimulus space 
    X_prime_trn_t = [ww.dot(vv).dot(t_fs) for ww,vv,t_fs in zip(W_trn,Vt_trn,trn_fs)]
    #S1_prime_trn_t = np.dot(np.dot(w1_alpha_trn, Vt1_trn), trn_stim_A.T) #w1_alpha_trn = 1200x1200, Vt1_trn = 1200x1200, trn_stim_A.T = 1200x3737
    Xcomb_prime_trn_t = np.vstack(X_prime_trn_t)

    # Create & modify covariance matrix
    stim_cov_mat_r = X_prime_trn_t.dot(X_prime_trn_t.T) / float(num_train_points-1)
    cov_diag = np.sqrt(np.diag(stim_cov_mat_r))
    full_mat_cov_diag = np.tile(cov_diag, [cov_diag.shape[0], 1])
    # re-do w/ simpler syntax?
    all_divisor = np.multiply(full_mat_cov_diag.T, full_mat_cov_diag) 
    corr_mat_r = np.divide(stim_cov_mat_r, all_divisor)

    ### --- Clean up the correlation matrix to have zeros where we know they exist and use that data to set a threshold --- ###
    idx_ct = np.cumsum([0]+n_abc)
    idxs = [(a,b) for a,b in zip(idx_ct[:-1],idx_ct[1:])]
    # Block diagonal components of covariance matrix
    for n,(ii,jj) in zip(n_abc,idxs):
        corr_mat_r[ii:jj] = np.eye(n)
    # Off-diagonal elements: ignore for now? 
    #for i1,i2 in zip(idxs[:-1],idxs[1:]):
    #    (ii,jj),(kk,ll) = i1,i2
        

    # ##### --- WORKING HERE - SEE IPYTHON NOTEBOOK --- #########
    # upper_right_corr = np.ravel(corr_mat_r[0:nA, nA:])
    # middle_right_corr = np.ravel(corr_mat_r[nA:(nA+nB),(nA+nB):])
    # right_corr = np.hstack([upper_right_corr, middle_right_corr])
    # s_right_corr = argsort(right_corr)
    # # WTF is this?
    # #corr_cutoff = 954 # WH magic number; something to do with the fact that it's needless to have 
    #                 # ALL the block-diagonal diagonals, since we have limited data
    # #goodcorrs_idx = np.hstack([s_right_corr[0:corr_cutoff], s_right_corr[-1:-(corr_cutoff+1):-1]])

    # new_right_corrs = np.squeeze(np.zeros([s_right_corr.shape[0],1]))
    # #new_right_corrs[goodcorrs_idx] = right_corr[goodcorrs_idx]
    # new_upper_right_corrs = np.reshape(new_right_corrs[0:(nB+nC)*nA],[nA,nB+nC])
    # new_lower_left_corrs = new_upper_right_corrs.T
    # new_middle_right_corrs = np.reshape(new_right_corrs[(nB+nC)*nA:],[nB,nC])
    # new_middle_left_corrs = new_middle_right_corrs.T

    # ##NEED TO CHANGE THIS: REMOVE HARDCODED MATRIX SIZES
    # new_corr_mat_r = copy.copy(corr_mat_r)
    # new_corr_mat_r[0:nA, nA:]= new_upper_right_corrs
    # new_corr_mat_r[nA:(nA+nB), (nA+nB):] = new_middle_right_corrs
    # new_corr_mat_r[(nA+nB):, nA:(nA+nB)]= new_middle_left_corrs # More like bottom middle
    # new_corr_mat_r[nA:,0:nA] = new_lower_left_corrs # 
    # new_corr_mat_r[0:nA,0:nA]= np.identity(nA)
    # new_corr_mat_r[nA:(nA+nB), nA:(nA+nB)] = np.identity(nB)
    # new_corr_mat_r[(nA+nB):,(nA+nB):] = np.identity(nC)

    #perform eigenvalue decomposition (WHAT FOR? delete this?)
    #w, v = np.linalg.eigh(new_corr_mat_r)
    # Invert modified covariance matrix
    #corr_r_inv = np.linalg.inv(new_corr_mat_r)
    corr_r_inv = np.linalg.inv(corr_mat_r)
    #for 
    ##create filter
    dot1 = np.dot(X_prime_trn_t, trn_data) #precompute for speed
    dot2 = np.dot(corr_r_inv, dot1) #precompute for speed
    # Weights
    h_123_prime = np.divide(dot2, (float(num_train_points-1)))


    ##create estimated responses from training data
    #r_hat = np.dot(X_prime_trn_t.T, h_123_prime) # not usually done...
    #if do_pred:
    #validation set results
    val_stim_A_prime = np.dot(np.dot(w1_alpha_r, Vt1_r), val_stim_A.T)
    val_stim_B_prime = np.dot(np.dot(w2_alpha_r, Vh2_r), val_stim_B.T)
    val_stim_C_prime = np.dot(np.dot(w3_alpha_r, Vh3_r), val_stim_C.T)
    #S1_prime = S1_prime[0:200,:]
    #S2_prime = S2_prime[0:200,:]

    S123_val_prime_t = np.vstack([val_stim_A_prime, val_stim_B_prime, val_stim_C_prime])

    #create validation set correlations
    r_hat_val = np.dot(S123_val_prime_t.T, h_123_prime)


    #look at performance
    valcorr = _sutils.column_corr(r_hat_val, val_data)
    out = dict(
        #weights=wt,
        #alphas=alphas,
        #n_sig_vox_byalpha=n_sig_vox_byalpha,
        cc=valcorr
        )
    return out


### --- Alex functions, keep / get rid of... --- ###

def ridge_AH(trn_fs, val_fs, trn_data, val_data, alphas, rval_data=None, rval_fs=None, saveallwts=True,
          stop_early=False, dtype=np.single, corrmin=0.2, singcutoff=1e-10):
    """Ridge regresses [trn_fs] onto [trn_data] for each ridge parameter in [alpha].  Returns the fit
    linear weights for each alpha, as well as the distributions of correlations on a held-out test
    set ([val_fs] and [val_data]).  Note that these should NOT be the "real" held-out test set, only a
    small test set used to find the optimal ridge parameter.
    
    If an [rval_data] and [rval_fs], or 'real' val_data and val_fs, are given, correlations on that dataset
    will be computed and displayed for each alpha.
    
    If [savallewts] is True, all weights will be returned.  Otherwise only the best weights will be
    returned.
    
    If [stop_early] is True, the weights and correlations will be returned as soon as the mean
    correlation begins to drop.  Does NOT imply early-stopping in the regularized regression sense.
    
    The given [dtype] will be applied to the regression weights as they are computed.

    Singular values less than [singcutoff] will be truncated.
    """
    ## Precalculate SVD to do ridge regression
    print "Doing SVD..."
    U,S,Vt = np.linalg.svd(trn_fs, full_matrices=False)
    ngoodS = np.sum(S>singcutoff)
    U = U[:ngoodS]
    S = S[:ngoodS]
    Vt = Vt[:ngoodS]
    print "Dropped %d tiny singular values.. (U is now %s)"%(np.sum(S<singcutoff), str(U.shape))
    val_datanorms = np.apply_along_axis(np.linalg.norm, 0, val_data) ## Precompute test response norms
    trn_corrs = [] ## Holds training correlations for each alpha
    Pcorrs = [] ## Holds test correlations for each alpha
    wts = [] ## Holds weights for each alpha
    bestcorr = -1.0 ## Keeps track of the best correlation across all alphas
    UR = np.dot(U.T, trn_data) ## Precompute this matrix product for speed
    for a in alphas:
        D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter 
        
        #wt = reduce(np.dot, [Vt.T, D, U.T, trn_data]).astype(dtype)
        wt = reduce(np.dot, [Vt.T, D, UR]).astype(dtype)
        pred = np.dot(val_fs, wt) ## Predict test responses
        prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
        #trn_corr = np.array([np.corrcoef(val_data[:,ii], pred[:,ii].ravel())[0,1] for ii in range(val_data.shape[1])]) ## Slowly compute correlations
        trn_corr = np.array(np.sum(np.multiply(val_data, pred), 0)).squeeze()/(prednorms*val_datanorms) ## Efficiently compute correlations
        trn_corr[np.isnan(trn_corr)] = 0
        trn_corrs.append(trn_corr)
        
        if saveallwts:
            wts.append(wt)
        elif trn_corr.mean()>bestcorr:
            bestcorr = trn_corr.mean()
            wts = wt
        
        print "Training: alpha=%0.3f, mean corr=%0.3f, max corr=%0.3f, over-under(%0.2f)=%d" % (a,
                                                                                                np.mean(trn_corr),
                                                                                                np.max(trn_corr),
                                                                                                corrmin,
                                                                                                (trn_corr>corrmin).sum()-(-trn_corr>corrmin).sum())
        
        ## Test alpha on real test set if given
        if rval_data is not None and rval_fs is not None:
            rpred = np.dot(rval_fs, wt)
            Pcorr = np.array([np.corrcoef(rval_data[:,ii], rpred[:,ii].ravel())[0,1] for ii in range(rval_data.shape[1])])
            Pcorr[np.isnan(Pcorr)] = 0.0
            print "Testing: alpha=%0.3f, mean corr=%0.3f, max corr=%0.3f" % (a, np.mean(Pcorr), np.max(Pcorr))
            Pcorrs.append(Pcorr)
            if sum(np.isnan(Pcorr)):
                raise Exception("nan correlations")

        ## Quit if mean correlation decreases
        if stop_early and trn_corr.mean()<bestcorr:
            break

    if rval_data is not None and rval_fs is not None:
        return wts, trn_corrs, Pcorrs
    else:
        return wts, trn_corrs

def ridge_corr(trn_fs, val_fs, trn_data, val_data, alphas, normalpha=False, dtype=np.single, corrmin=0.2, singcutoff=1e-10):
    """
    Fits only alpha parameter (through n_splits cross-validation splits of data)
    
    AH Notes: 
    Uses ridge regression to find a linear transformation of [trn_fs] that approximates [trn_data].
    Then tests by comparing the transformation of [val_fs] to [val_data]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. Note that the regression weights are NOT returned.

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
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
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
    ## Calculate SVD of stimulus matrix
    print "Doing SVD..."
    try:
        U,S,Vt = np.linalg.svd(trn_fs, full_matrices=False)
    except np.linalg.LinAlgError, e:
        print "NORMAL SVD FAILED, trying more robust dgesvd.."
        from .svd_dgesvd import svd_dgesvd
        U,S,Vt = svd_dgesvd(trn_fs, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S>singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vt = Vt[:ngoodS]
    print "Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape))

    ## Normalize alpha by the Frobenius norm
    frob = np.sqrt((S**2).sum()) ## Frobenius!
    #frob = S.sum()
    print "Training stimulus has Frobenius norm: %0.03f"%frob
    if normalpha:
        nalphas = alphas * frob
    else:
        nalphas = alphas

    ## Precompute some products for speed
    UR = np.dot(U.T, trn_data) ## Precompute this matrix product for speed
    PVh = np.dot(val_fs, Vt.T) ## Precompute this matrix product for speed
    
    val_datanorms = np.apply_along_axis(np.linalg.norm, 0, val_data) ## Precompute test response norms
    trn_corrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        #D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter 
        D = S/(S**2+na**2) ## Reweight singular vectors by the (normalized?) ridge parameter
        
        pred = np.dot(_utils.mult_diag(D, PVh, left=False), UR) ## Best? (1.75 seconds to prediction in test)
        
        prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
        #trn_corr = np.array([np.corrcoef(val_data[:,ii], pred[:,ii].ravel())[0,1] for ii in range(val_data.shape[1])]) ## Slowly compute correlations
        trn_corr = np.array(np.sum(np.multiply(val_data, pred), 0)).squeeze()/(prednorms*val_datanorms) ## Efficiently compute correlations
        trn_corr[np.isnan(trn_corr)] = 0
        trn_corrs.append(trn_corr)
        
        print "Training: alpha=%0.3f, mean corr=%0.3f, max corr=%0.3f, over-under(%0.2f)=%d" % (a,
                                                                                                np.mean(trn_corr),
                                                                                                np.max(trn_corr),
                                                                                                corrmin,
                                                                                                (trn_corr>corrmin).sum()-(-trn_corr>corrmin).sum())    
    return trn_corrs



def ridge_boot(trn_fs, trn_data, val_fs, val_data, alphas, nboots, chunklen, n_chunks, dtype=np.single, corrmin=0.2):
    """Uses ridge regression with a bootstrapped held-out set to get a single optimal alpha values for all voxels.
    [n_chunks] random chunks of length [chunklen] will be taken from [trn_fs] and [trn_data] for each regression
    run. [nboots] total regression runs will be performed.
    """
    n_resp, n_voxels = trn_data.shape
    bestalphas = np.zeros((nboots, n_voxels))  ## Will hold the best alphas for each voxel

    Rcmats = []
    for bi in range(nboots):
        print "Selecting held-out test set.."
        allinds = range(n_resp)
        indchunks = zip(*[iter(allinds)]*chunklen)
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:n_chunks]))
        notheldinds = list(set(allinds)-set(heldinds))
        
        trn_fs_split = trn_fs[notheldinds,:]
        val_fs_split = trn_fs[heldinds,:]
        trn_data_split = trn_data[notheldinds,:]
        val_data_split = trn_data[heldinds,:]
        
        ## Run ridge regression using this test set
        Rwts, trn_corrs = ridge_AH(trn_fs_split, val_fs_split, trn_data_split, val_data_split, alphas,
                             saveallwts=False, dtype=dtype, corrmin=corrmin)
        
        Rcmat = np.vstack(trn_corrs)
        Rcmats.append(Rcmat)
        #bestainds = np.array(map(np.argmax, Rcmat.T))
        #bestalphas[bi,:] = alphas[bestainds]
    
    print "Finding best alpha.."
    ## Find best alpha for each voxel
    cc = np.dstack(Rcmats)
    meanbootcorr = cc.mean(2).mean(1)
    bestalphaind = np.argmax(meanbootcorr)
    alpha = alphas[bestalphaind]
    print "Best alpha = %0.3f"%alpha
    
    ## Find weights for each voxel
    U,S,Vt = np.linalg.svd(trn_fs, full_matrices=False)
    UR = np.dot(U.T, np.nan_to_num(trn_data))
    pred = np.zeros(val_data.shape)
    wt = reduce(np.dot, [Vt.T, np.diag(S/(S**2+alpha**2)), UR])
    pred = np.dot(val_fs, wt)

    ## Find test correlations
    nnpred = np.nan_to_num(pred)
    cc = np.nan_to_num(np.array([np.corrcoef(val_data[:,ii], nnpred[:,ii].ravel())[0,1] for ii in range(val_data.shape[1])]))
    return wt, cc
