"""
CCA analysis code, for two matrices.

From vanilla_cca.py which was kindly provided by Natalia Bilenko

ML added loop to find optimal regularization paramters

"""

import numpy as np
from sklearn import cross_validation
from scipy.linalg import eigh

default_kappas = np.logspace(-3,1,10)

def run_cca(data, kappa=0, numeig=None, kernelcca=True,ktype='linear',returncorrs=False):
    '''CCA analysis with specified kappa for regularization and optional kernel

    Set up and solve the eigenproblem for the data in kernel and specified kappa

    Parameters
    ----------
    data : list of arrays
        matrices to be cross-decomposed,  of size n x p, n x q
        (first dim must be the same!)
    kappa : non-negative real scalar, optional
        hyper-parameter for regularization
    numeig : positive integer, optional (default = minimum of p,q)
        number of eigenvectors of the covariance matrix to keep
        (= number of canonical correlates to keep)
    kernelcca : bool, optional
        Whether to use kernel trick or not
    returncorrs : bool, optional
        return correlations between X_projected, Y_projected as second output

    Other Parameters
    ----------------
    ktype : string, 'linear' only for now
        type of kernel to compute if kernelcca==True

    Returns
    -------
    W : list of arrays
        Transformation matrices to project data matrices into canonical 
        (correlated) space.

    '''
    # Do / don't do kernel
    if kernelcca:
        kernel = [make_kernel(d) for d in data]
    else:
        kernel = [d.T for d in data]

    nTs = [k.shape[0] for k in kernel]
    numeig = min(nTs) if numeig is None else numeig

    # Get the kernel auto- and cross-covariance matrices between X and Y
    crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]

    # Allocate LH and RH:
    LH = np.zeros((np.sum(nTs), np.sum(nTs)))
    RH = np.zeros((np.sum(nTs), np.sum(nTs)))

    # Fill the left and right sides of the eigenvalue problem
    for i in range(len(kernel)):
        RH[int(np.sum(nTs[:i])):int(np.sum(nTs[:i+1])), int(np.sum(nTs[:i])):int(np.sum(nTs[:i+1]))] = crosscovs[i*(len(kernel)+1)] + kappa*np.eye(nTs[i])
        for j in range(len(kernel)):
            if i !=j:
                LH[int(np.sum(nTs[:j])):int(np.sum(nTs[:j+1])), int(np.sum(nTs[:i])):int(np.sum(nTs[:i+1]))] = crosscovs[len(kernel)*j+i]

    LH = (LH+LH.T)/2.
    RH = (RH+RH.T)/2.

    r, Vs = eigh(LH, RH)

    if kernelcca:
        W = []
        for i in range(len(kernel)):
            W.append(Vs[int(np.sum(nTs[:i])):int(np.sum(nTs[:i+1])), :numeig])
        tcorrs = kcca_recon(data, W, corronly=True)
        tc = [t[0, 1] for t in tcorrs]
        # Sort transformation matrices by correlation coefficient
        i = np.argsort(tc)[::-1]
        W = [c[:, i[:numeig]] for c in W]
    else:
        # Get vectors for each dataset
        r[np.isnan(r)] = 0
        rindex = np.argsort(r)[::-1]
        r = r[rindex]
        W = []
        Vs = Vs[:, rindex]
        rs = np.sqrt(r[:numeig]) # these are not correlations for kernel CCA with regularization
        for i in range(len(kernel)):
            W.append(Vs[int(np.sum(nTs[:i])):int(np.sum(nTs[:i+1])), :numeig])
    if returncorrs:
        if kernelcca:
            return W, tcorrs
        else:
            return W, rs
    return W

def make_kernel(d,ktype='linear'):
    '''Makes a linear inner product kernel for data d
    '''
    if ktype=='linear':
        d = np.nan_to_num(d)
        cd = d-d.mean(0)
        kernel = np.dot(cd,cd.T)
        kernel = (kernel+kernel.T)/2.
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    else:
        raise NotImplementedError('Unknown kernel type!')
    return kernel

def kcca_recon(data, W, corronly=False):
    nT = data[0].shape[0]
    # Get canonical variates and CCs
    ws = [np.dot(x[0].T, x[1]) for x in zip(data, W)]
    ccomp = [np.dot(x[0].T, x[1]) for x in zip([d.T for d in data], ws)]
    corrs = listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return ws, corrs

def listcorr(a):
    '''Returns pairwise row correlations for all items in array as a list of matrices
    '''
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j>i:
                corrs[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0,1]) for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs

def run_corrs(a,b):
    """Run correlations between all pairs of columns for two matrices, return as a vector"""
    corrs = np.array([np.nan_to_num(np.corrcoef(ai,bi)[0,1]) for ai,bi in zip(a.T,b.T)])
    return corrs

def run_cca_cv(data, numCC=None, kappas=default_kappas, kernelcca=True, ktype='linear',
        n_folds=10, returncorrs=False,is_verbose=False):
    """Regularized CCA w/ k-fold cross validation to find ridge parameter

    Parameters
    ----------
    data : list of arrays 
        data matrices to be cross-decomposed
    numCC : positive int or list of positive ints, optional
        Number of eigenvectors to keep. Potentially tries many, if
        a list is given. Defaults to all CCs (`data[0].shape[1]` if
        kernelcca is False, `data[0].shape[0]` if kernelcca is True)
    kappas : list or 1d array, optional
        Regularization parameters to attempt. Defaults to value 
        specified in cca_simple.py, logspace(-3,1,10)
    kernelcca : bool, optional
        Whether to use kernel CCA or not.
    n_folds : positive int, optional
        Number of folds of the data to use.
    returncorrs : bool, optional
        Whether to return correlation coefficients, kappas, and 

    """
    nT = data[0].shape[0]
    # Deal with option to keep multiple numbers of CCs
    if numCC is None:
        # keep all CCs by default
        numCC = data[0].shape[1]
    if np.isscalar(numCC):
        # Make sure it's a list
        numCC = [numCC]
    # Split data into chunks
    folds = cross_validation.KFold(nT,n_folds,shuffle=False)
    # Preallocate matrix to store correlation coefficients
    cc = np.zeros((n_folds,len(numCC),len(kappas)))
    # k folds of data
    for irpt,(ti,vi) in enumerate(folds):
        if is_verbose: print("-- Data fold %d/%d --"%(irpt+1,n_folds))
        # (optionally) loop over different #s of eigenvalues/CCs
        for incc, numeig in enumerate(numCC):
            if is_verbose: print("Training CCA w/ %d eigenvectors"%numeig)
            # test different kappas
            for ik,k in enumerate(kappas):
                ws = run_cca([d[ti] for d in data],numeig=numeig,kappa=k,kernelcca=kernelcca,ktype=ktype,returncorrs=False)
                # Create projections of withheld validation data into canonical space
                xv_cc,yv_cc = [d[vi].dot(w) for d,w in zip(data,ws)]
                cctmp = run_corrs(xv_cc,yv_cc)
                # Take mean of first third of CCs to evaluate goodness of fit
                first_n_ccs = int(len(cctmp)/3) if len(cctmp)>9 else len(cctmp)
                cc[irpt,incc,ik] = cctmp[:first_n_ccs].mean()
            if is_verbose:
                print(('%-12s:'%'kappa')+('%-7s'*len(kappas))%tuple(['%.3f'%kk for kk in kappas]))
                print(('%-12s:'%'corr. coef')+('%-7s'*len(kappas))%tuple(['%.3f'%kk for kk in np.squeeze(cc[irpt,incc,:])]))
    # Choose optimal regression parameter (kappa), # of eigenvectors
    # avg. over repeats, left w/ mean corr by # eigenvectors, kappa
    mean_corrs = cc.mean(0) 
    best_cidx, best_kidx = np.where(mean_corrs == mean_corrs.max())
    k,numeig = kappas[best_kidx],numCC[best_cidx]
    W, cc = run_cca(data, kappa=k, numeig=numeig, kernelcca=kernelcca,ktype='linear',returncorrs=True)
    if returncorrs:
        return W,cc,mean_corrs,k
    else:
        return W

def run_cca_boot(data, numCC=None, kappas=default_kappas, kernelcca=True, ktype='linear',
        chunklen=10, bootstraps=10, returncorrs=False):
    """Regularized CCA w/ bootstrap to find ridge params"""
    raise NotImplementedError('Still working on option for kernel/not')
    nT = data[0].shape[0]
    numCC = [nT] if numCC is None else numCC
    nchunks = int(0.2*nT/chunklen)
    allinds = range(nT)
    indchunks = zip(*[iter(allinds)]*chunklen)
    corr_matrix = np.zeros((len(numCC), len(kappas), bootstraps))
    for cci, numeig in enumerate(numCC):
        for ki, kappa in enumerate(kappas):
            for bootstrap in range(bootstraps):
                random.shuffle(indchunks)
                heldinds = list(itertools.chain(*indchunks[:nchunks]))
                notheldinds = list(set(allinds)-set(heldinds))
                comps = kcca([d[notheldinds] for d in data], kappa, numeig, order = True)
                ws, cs = kcca_valrecon([d[notheldinds] for d in data], [d[heldinds] for d in data], comps)
                pred, corrs = kcca_predict([d[heldinds] for d in data], ws)
                corr_matrix[cci, ki, bootstrap] = np.mean([c.mean() for c in corrs])

    mean_corrs = corr_matrix.mean(2)
    best_cidx, best_kidx = np.where(mean_corrs == mean_corrs.max())
    best_numCC = numCC[best_cidx]
    best_kappa = kappas[best_kidx]
    comps = kcca(data, best_kappa, best_numCC, order = True)
    return comps


