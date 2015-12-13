# # Stats utils

# import numpy as np
# import itertools as itools
# import copy
# from scipy.interpolate import interp1d
# import scipy.stats as _stats
# from ..utils import wrap,unwrap,as_list
# import warnings

# def column_corr(A,B,dof=0):
#     """Efficiently compute correlations between columns of two matrices
    
#     Does NOT compute full correlation matrix btw `A` and `B`; returns a 
#     vector of correlation coefficients. FKA ccMatrix."""
#     # zs = lambda x: (x-np.nanmean(x,axis=0))/np.nanstd(x,axis=0,ddof=dof)
#     zs = lambda x: np.true_divide((x-np.nanmean(x,axis=0)), np.nanstd(x,axis=0,ddof=dof))
#     rTmp = np.nansum(zs(A)*zs(B),axis=0)
#     n = A.shape[0]
#     # make sure not to count nans
#     nNaN = np.sum(np.logical_or(np.isnan(zs(A)),np.isnan(zs(B))),0)
#     n = n - nNaN
#     # r = rTmp/n
#     r = np.true_divide(rTmp, n)
#     return r

# def corr_mean(data,*args,**kwargs):
#     """Take the mean of an array of correlation coefficients, first z-scoring them

#     Uses np.arctanh to Fischer z-transform an array of correlations, then passes that
#     array to np.nanmean
#     """
#     dataz = np.arctanh(data)
#     return np.nanmean(data,*args,**kwargs)

# def ci(x,a,n=1000,ax=0):
#     """Get confidence interval for mean, w/ n bootstrap samples
    
#     DOES NOT WORK FOR ARRAYS > 2 DIMENSIONS

#     Parameters
#     ----------
#     a is from 0-100 (95 for 95% confidence interval)
#     output range should GROW with a higher value for `a`
#     """
#     if isinstance(x,(list,tuple)):
#         x = np.array(x)
#     if np.ndim(x)==1:
#         x = np.reshape(x,(len(x),1))
#     if ax > 1:
#         raise Exception("Not working for arrays > 2D yet!")
#     elif ax==1:
#         x = x.T
#     # Need to fix here for multi-dim arrays!
#     LL = x.shape[0]
#     #p_ax = (ax,)+tuple([d for d in range(x.ndim) if not d==ax])
#     cc = [[np.nanmean(np.random.choice(xx,LL,replace=True)) for xx in x.T] for c in range(n)]
#     cc = np.vstack(cc)
#     a = 100.-a
#     pct = np.percentile(cc,[a/2.,100-a/2.],axis=ax)
#     return pct


# def compute_R2(data,bias_corr=True):
#     """Compute coefficient of determination ($R^2$) for a dataset.

#     Parameters
#     ----------
#     data : 3D array
#         repeats x time x samples [voxels] 
#         (TO DO: Update to handle 2D time x repeats too)
#     bias_corr : bool
#         whether to do bias correction (centers distribution of random data on 0)
    
#     Returns
#     -------
#     r2 : array
#         Coefficient of determination (R^2) for data, one per sample (voxel)
#     """
#     n_rpts,n_t,n_samples = data.shape
#     vVar1 = (data-np.nanmean(data,0)).reshape(n_rpts*n_t,n_samples).var(0)
#     vVar2 = np.var(np.reshape(data,(n_rpts*n_t,n_samples)),0)
#     # EV = 1 - (variance w/ mean subtracted off) / (total variance)
#     ev = 1-vVar1/vVar2
#     # Bias correction (always)
#     if bias_corr:
#         ev = ev-(1-ev)/n_rpts
#     return ev

# ### THESE TWO FUNCTIONS ( /|\ \|/ ) SHOULD TAKE IDENTICAL INPUTS. 

# def compute_mpwc(data):
#     """Compute mean pairwise correlation between repeats
    
#     Parameters
#     ----------
#     data : array-like
#         repeated data; should be (repeats x time x samples [voxels]) 
    
#     Returns
#     -------
#     cc : vector
#         correlation per sample (voxel)

#     TO DO
#     -----
#     Make this (optionally) more memory-efficient, with correlations
#     computed in chunks
#     """
#     n_rpts,n_t,n_samples = data.shape
#     # Get all pairs of data
#     pairs = [p for p in itools.combinations(np.arange(n_rpts),2)]
#     # Preallocate
#     r = np.nan*np.zeros((n_samples,len(pairs)))
#     for p,(a,b) in enumerate(pairs):
#         r[:,p] = column_corr(data[a],data[b])
#     cc = np.nanmean(r,1);
#     return cc

# def t_contrast(B,C,X,Y,is_two_sided_p=True):
#     """Compute t contrast for a set of weights

#     Only (?) appropriate for weights fit using OLS regression.

#     Largely borrowed from M. Brett's course on practical fMRI, http://perrin.dynevor.org/glm_intro.html
#     Parameters
#     ----------
#     B : array-like
#         weight matrix, features x voxels
#     C : array-like
#         Contrast vector (often 1s, 0s, -1s), as long as total number of weights
#     X : array-like 
#         Stimulus matrix
#     Y : array-like
#         data that were originally fit

#     Returns
#     -------
#     t : array-like
#         t statistic

#     Notes
#     -----
#     $t = \frac{c^T \hat\beta}{\sqrt{\hat{\sigma}^2 c^T (X^T X)^+ c}}$

#     where $\hat{\sigma}^2$ is the estimate of variance in the residuals, 
#     and $(X^T X)^+$ is the Penrose pseudo-inverse of $X^T X$.

#     This is OLS estimation; we assume the errors to have independent
#     and identical normal distributions around zero for each $i$ in 
#     $\epsilon_i$ (i.i.d).

#     TO DO
#     -----
#     This may need breaking up into chunks for large contrasts
#     """
#     # Make sure X, Y, C are all arrays
#     B = np.asarray(B)
#     X = np.asarray(X)
#     C = np.atleast_2d(C)
#     Y = np.asarray(Y)
#     # The fitted values - this may be an expensive step
#     fitted = X.dot(B)
#     # Residual sum of squares
#     RSS   = ((Y - fitted)**2).sum(axis=0)
#     # Degrees for freedom is the number of observations n minus the number
#     # of independent regressors we have used.  If all the regressor columns
#     # in X are independent then the (matrix rank of X) == p
#     # (where p the number of columns in X). If there is one column that can
#     # expressed as a linear sum of the other columns then (matrix rank of X)
#     # will be p - 1 - and so on.
#     df =  X.shape[0] - np.linalg.matrix_rank(X)
#     # Mean residual sum of squares
#     MRSS  = RSS / df
#     # calculate bottom half of t statistic
#     SE    = np.sqrt(MRSS * C.dot(np.linalg.pinv(X.T.dot(X)).dot(C.T)))
#     t     = C.dot(B)/SE
#     p = t2pval(t,df,is_two_sided=is_two_sided_p)
#     return t, df, p 


# def t2pval(t,dof,is_two_sided=False):
#     """Convert t value to p value give dof"""
#     if is_two_sided:
#         t=abs(t); # use |t| for two sided P-value
#         p=2*(1-_stats.t.cdf(t,dof));
#     else:
#         p=2*(1-_stats.t.cdf(t,dof))/2; # divde by two for one-tailed test
#     return p


# def distr2thr(dst,p_lev=.001,thr_start=0,thr_incr=.001,is_two_sided=True):
#     """Convert a threshold value given an empirical distribution of data
    
#     Useful for finding a threshold at a particular `p_lev` given the results 
#     of a bootstrapping analysis
    
#     Parameters
#     ----------
#     dst : array-like
#         empirical probability distribution of data values
#     p_lev: scalar
#         desired p threshold
    
#     """
#     thr = thr_start
#     max_iter = 1000000
#     if is_two_sided:
#         p_lev/=2.
#         thr_lo = copy.copy(thr)
#         iiter = 0
#         while (distr2p(dst,thr_lo,is_two_sided=is_two_sided)>p_lev) and iiter<max_iter:
#             thr_lo-=thr_incr
#             iiter+=1
#     iiter = 0
#     while (distr2p(dst,thr,is_two_sided=is_two_sided)>p_lev) and iiter<max_iter:
#             thr+=thr_incr
#             iiter+=1
#     if iiter==max_iter:
#         raise Exception("Max iterations reached!")
#     if is_two_sided:
#         return thr_lo,thr
#     else:
#         return thr
    
# def distr2p(dst,data,is_two_sided=True,posneg=True):
#     """Compute p for each value in data given empirical distribution dst
#     ASSUMES distribution is centered on zero 
#     FOR NOW, only 1-D `data` is possible. `dst` should be 1D as well."""
#     data = as_list(data)
#     if not np.allclose(np.mean(dst),0,atol=1e-3):
#         warnings.warn('distr2p assumes empirical distribution has mean=0! ASSUMPTION VIOLATED!')
#     p = np.array([np.mean(np.abs(dst)>np.abs(x)) for x in data])
#     if not is_two_sided:
#         p /= 2.
#     return p

# def makeUniform(*args,**kwargs):
#     warnings.warn('Deprecated! use make_uniform!')
#     return make_uniform(*args,**kwargs)

# def make_uniform(data,CDFres=1000,xdp=1):
#     """Convert data to the probability of each data point assuming a uniform distribution.
    
#     Useful as the first step of Gaussianizing data.
    
#     Example:
#     x = rand(1000,1);
#     xU = makeUniform(x); # Already almost uniform b/c of "rand"; make it exactly so  
#     xN = norminv(xU); # Take inverse normal distribution (given probabilities) 
#     hist(x,25); figure; hist(xN,25);
#     t = 1:1000;
#     figure; plot(t,x,'b',t,xN,'ro');
    
#     Modified slightly from Dustin Stansbury's class method
#     gaussianize.makeUniform by ML on 2013.03.06
#     """
#     # TRANSFORM DATA TO UNIFORM DISTRIBUTION
#     # BASED ON CUMULATIVE DISTRIBUTION FUNCTION
#     maxx = np.max(data); minn = np.min(data)
#     xOffset = (xdp/100.)*np.abs(maxx - minn)
#     xGrid = np.linspace(minn,maxx,np.sqrt(data.size)+1)
#     binC = (xGrid[:-1]+xGrid[1:])/2.

#     [counts,binE] = np.histogram(data,xGrid); # bin edges

#     # CALCULATE CUMULATIVE DISTRIBUTION
#     CDF = np.cumsum(counts).astype('float64')
#     N = np.max(CDF);
#     CDF = CDF/N*(1.-1./N);

#     # ENSURE SUPPORT AT EXTREMUM OF CDF
#     dX = np.mean(np.diff(binC))/2;
#     xCDF = np.hstack([minn-xOffset,minn,(binC + dX), maxx + xOffset + dX])
#     yCDF = np.hstack([0,1./N,CDF,1]);

#     # UPSAMPLE CDF FOR LOOKUP/TRANSFORM
#     xUpsample = np.linspace(xCDF[0],xCDF[-1],CDFres)

#     spl = interp1d(xCDF,yCDF)  
#     cdfUpsample = spl(xUpsample)
#     cdfUpsample = ensureMonotonic(cdfUpsample)

#     # SCALE TO ENSURE MAX OF CDF IS ONE
#     cdfUpsample = cdfUpsample/np.max(cdfUpsample)

#     # TRANSFORM DATA TO UNIFORM DISTRIBUTION
#     spl2 = interp1d(xUpsample,cdfUpsample)
#     dU = spl2(data)
#     return dU

# def ensureMonotonic(x):
#     warnings.warn('Deprecated! Use ensure_monotonic!')
#     return ensure_monotonic(x)

# def ensure_monotonic(x):
#     """
#     Ensures the entry of data are monotonically increasing (IDKWTF that means, this is from Dustin)
#     """
#     x = x.astype('float64')
#     for iI in range(1,x.size):
#         if x[iI] <= x[iI-1]:
#             if np.abs(x[iI-1]) > 1e-14:
#                 x[iI] = x[iI-1] + 1e-14
#             elif x[iI-1] == 0:
#                 x[iI] = 1e-80
#             else:
#                 x[iI] = x[iI-1] + 10^(np.log10(np.abs(x[iI-1])))
#     return x

# def r2pval(r,n,is_two_sided=True):
#     """Convert r values to p values by standard t distribution assumption.

#     Parameters
#     ----------

#     """
#     r = np.array(as_list(r))
#     if n < 3:
#         raise Exception('n < 3');
#     r[r==1.] = np.nan
#     t=np.sqrt(n-2)*r/(np.sqrt(1-r*r));  # this is t with n-2 degrees of freedom
#     p = t2pval(t,n-2)
#     # if is_two_sided:
#     #     t=abs(t); # use |t| for two sided P-value
#     #     p=2*(1-_stats.t.cdf(t,n-2));
#     # else:
#     #     p=2*(1-_stats.t.cdf(t,n-2))/2; # divde by two for one-tailed test
#     # Slightly sketch - could turn errors into highly unlikely (p=0) values...
#     p[np.isnan(p)] = 0
#     return p

# def pval2r(pval,n,is_two_sided=True,r_res=.001):

#     r = 0.;
#     while True:
#         r += r_res
#         ptemp = r2pval(r,n,is_two_sided);
#         if ptemp <= pval:
#             break
#     return r

# def fdr_correct2(pval,thres):
#     """Find the fdr corrected p-value thresholds
    
#     This is an abortive attempt to (re-) compute p values for ALL values
#     in the distribution `pval`, instead of just computing a single threshold...
#     Not sure if this is exactly what fdr correction was intended to do. Thus
#     abandoned.


#     Parameters
#     ----------
#     pval : array-like
#         vector of p-values
#     thres : scalar
#         FDR level (e.g. .05)
    
#     Returns 
#     -------
#     pID :
#         p-value threshold based on independence or positive dependence
#     pN : 
#         Nonparametric p-val thres
#     """
#     # remove NaNs, keep index
#     nan_idx = np.isnan(pval)
#     p = pval[nan_idx==False]
#     # 
#     p = np.sort(p)
#     V = np.float(len(p))
#     I = np.arange(V) + 1

#     cVID = 1
#     cVN = (1/I).sum()

#     th1 = np.nonzero(p <= I/V*thres/cVID)[0]
#     th2 = np.nonzero(p <= I/V*thres/cVN)[0]
#     if len(th1)>0:
#         pID = p[th1.max()]
#     else:
#         pID = -np.inf
#     if len(th2)>0:
#         pN =  p[th2.max()]
#     else:
#         pN = -np.inf

#     return pID, pN

# def fdr_correct(pvals,p_thresh):
#     """Find the fdr corrected p-value thresholds
    
#     Parameters
#     ----------
#     pval : array-like
#         vector of p-values
#     thres : scalar
#         FDR level (e.g. .05)
    
#     Returns 
#     -------
#     pID :
#         p-value threshold based on independence or positive dependence. This is 
#         the value at which the probability is (actually) p_thresh for this set of 
#         p values
#     pN : 
#         Nonparametric p-val thres. IDKWTF this is.
#     """
#     # remove NaNs
#     p = pvals[np.nonzero(np.isnan(pvals)==False)[0]]
#     p = np.sort(p)
#     V = np.float(len(p))
#     I = np.arange(V) + 1

#     cVID = 1
#     cVN = (1/I).sum()

#     th1 = np.nonzero(p <= I/V*p_thresh)[0]
#     th2 = np.nonzero(p <= I/V*p_thresh/cVN)[0]
#     if len(th1)>0:
#         pID = p[th1.max()]
#     else:
#         pID = -np.inf
#     if len(th2)>0:
#         pN =  p[th2.max()]
#     else:
#         pN = -np.inf

#     return pID, pN


# def predict(dataset,model,ppstim,order=None):
#     """Predict responses on a data set given a fit model.

#     For now, `DataSet` is assumed to be SEPARATE from the data used to 
#     fit the model. 

#     FOR NOW: ONLY ARRAYS. Disregard parameters below; full of lies.
#     * changed
#     Parameters
#     ----------
#     dataset : dict | array-like
#         Query parameters for an fMRI_DataSet instance in the database, 
#         OR a 2-D array of time x voxels 
#         OR a 4-D array (IFF `mask` is provided)
#     model : fMRI_FitModel instance | array-like
#         EITHER an fMRI_FitModel instance (contains info about the model, 
#         preprocessed stim, etc; must have weights loaded)
#         OR a channels x voxels matrix of weights. If `model` is an array,
#         `ppstim` must be provided as well.
#     db : docdb database client | default=None
#         To query the database, if necessary
#     mask : dict | array-like
#         Query parameters for a mask instance in the database, 
#         OR a 3-D array of boolean values for voxels to keep in 
#         4-D dataset
#     """

#     pred = ppstim.dot(model)
#     if not order is None:
#         pred = wrap(pred,order)
#         if dataset.shape[0]!=pred.shape[0]:
#             # ...
#             dataset = wrap(dataset,order)
#     cc = ccMatrix(pred,dataset)
#     return cc


# #############################
# ### --- Bootstrapping --- ###
# #############################

# def bootstrap_chance_diff_shuf(preds,data, n_resamples=10000,print_every=1000):
#     """Computes chance distribution correlations by shuffling extant predictions
    
#     A reasonable first-pass way to compute probabilities; not as good as bootstrapping
#     the entire regression process (re-running regression n times with scrambled design
#     matrix row indices)

#     Parameters
#     ----------
#     preds : array-like
#         predictions; time x (voxels)
#     data : array-like
#         actual data; time x (voxels)
#     n_resamples : scalar
#         duh
#     print_every : scalar, < n_resamples
#         show progress by printing output line every `print_every` bootstrap iterations
#     """
#     # initialize p
#     p = []
#     # Figure out how many voxels in each ROI
#     nt = data.shape[0]
#     dstr = np.zeros((n_resamples,))
#     for i in range(n_resamples):
#         if i%print_every==0:
#             print(i)
#         # Shuffle indices for predictions / data
#         idx1 = np.random.permutation(nt)
#         idx2 = np.random.permutation(nt)
#         # arctanh is equivalent to Fischer's z score
#         cc = np.arctanh(column_corr(preds[idx1],data[idx2]))
#         dstr[i] = np.nanmean(cc)
#     return dstr


# def partition_variance(models,mnms=None,corr_thresh=-np.inf):
#     """

#     Parameters
#     ----------

#     """
#     model_names = [k for k in models.key() if not '+' in k]
#     if not mnms is None:
#         model_names = [m for m in model_names if m in mnms]
#     n_models = len(model_names) 
#     # Ugly, but this makes the syntax much easier to follow below
#     if n_models==2:
#         A,B = model_names
#     elif n_models==3:
#         A,B,C = model_names
#     elif n_models==4:
#         A,B,C,D = model_names
#     elif n_models==5:
#         A,B,C,D,E = model_names
#     else:
#         raise ValueError("partition_variance can't handle more than 5 models!")
#     # Restrict correlations to positive correlations
#     mods = {}
#     for vp in models.keys():
#         if isinstance(models[vp],dict):
#             CC = np.clip(models[vp]['cc'],0,np.inf)
#         else:
#             CC = np.clip(models[vp],0,np.inf)
#         mods[vp] = CC**2 # * np.sign(CC) # ? 
#     del CC
    

# def partition_variance2(models,mnms=('A','B'),varnm='cc',corr_thresh=-np.inf,keep_sign=False):
#     """Partition variance among all models

#     Parameters
#     ----------
#     models : dict
#         is a dict, with all combinations of models
#     """
#     from ..Classes.MappedClass import MappedClass
#     # Restrict correlations to positive correlations
#     mods = {}
#     for vp in models.keys():
#         if isinstance(models[vp],(dict,MappedClass)):
#             CC = np.clip(models[vp][varnm],0,np.inf)
#         else:
#             CC = np.clip(models[vp],0,np.inf)
#         mods[vp] = CC**2 * np.sign(CC) if keep_sign else CC**2 #mods[vp] = CC**2 # * np.sign(CC) # ? 
#     del CC
#     A,B = mnms
#     # Calculate different parts of Venn diagram: 
#     # Unique components for each individual model
#     Au = mods['%s+%s'%(A,B)]-mods[B]
#     Bu = mods['%s+%s'%(A,B)]-mods[A]

#     # Two-way intersections (not necessarily unique to these two parts)
#     ABi = mods[A] + mods[B] - mods['%s+%s'%(A,B)]
    
#     # Set explained variance below corr_thresh to zero
#     Au[Au<corr_thresh] = 0
#     Bu[Bu<corr_thresh] = 0
#     ABi[ABi<corr_thresh] = 0
    
#     # Output
#     data_partitions = {A+'u':Au, B+'u': Bu,A+B+'iu': ABi}
#     return data_partitions

# def partition_variance3(models,corr_thresh=-np.inf,varnm='cc',keep_sign=False):
#     # Restrict correlations to positive correlations
#     mods = {}
#     for vp in models.keys():
#         if isinstance(models[vp],dict):
#             CC = models[vp][varnm] if keep_sign else np.clip(models[vp][varnm],0,np.inf)
#         else:
#             CC = models[vp] if keep_sign else np.clip(models[vp],0,np.inf)
#         mods[vp] = CC**2 * np.sign(CC) if keep_sign else CC**2
#     del CC
#     # Maximum possible variance (if all mods are independent)
#     ABCmax = mods['A'] + mods['B'] + mods['C']
    
#     # Calculate different parts of Venn diagram: 
#     # Unique components for each individual model
#     Au = mods['A+B+C']-mods['B+C']
#     Bu = mods['A+B+C']-mods['A+C']
#     Cu = mods['A+B+C']-mods['A+B']

#     # Two-way intersections (not necessarily unique to these two parts)
#     ABi = mods['A'] + mods['B'] - mods['A+B']
#     ACi = mods['A'] + mods['C'] - mods['A+C']
#     BCi = mods['B'] + mods['C'] - mods['B+C']
    
#     # Three-way intersection (variance shared by all mods)
#     ABCi = (Au+Bu+Cu) + (ABi+ACi+BCi) - mods['A+B+C']
#     # The middle term above includes the common variance 3x
#     ABCi /= 2.0
    
#     # Exclude three-way intersection from two-way intersections
#     ABiu = ABi - ABCi
#     ACiu = ACi - ABCi
#     BCiu = BCi - ABCi
    
#     # Set explained variance below corr_thresh to zero
#     Au[Au<corr_thresh] = 0
#     Bu[Bu<corr_thresh] = 0
#     Cu[Cu<corr_thresh] = 0
#     ABiu[ABiu<corr_thresh] = 0
#     ACiu[ACiu<corr_thresh] = 0
#     BCiu[BCiu<corr_thresh] = 0
#     ABCi[ABCi<corr_thresh] = 0
    
#     # Output
#     data_partitions = {'Au':Au, 'Bu': Bu, 'Cu': Cu,'ABiu': ABiu, 'ACiu': ACiu, 'BCiu': BCiu, 'ABCi': ABCi}
#     return data_partitions
