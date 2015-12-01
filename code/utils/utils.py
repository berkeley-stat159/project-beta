
"""
These functions are meant to manipulate and reshape data, to facilitate bootstrapping, 


Statistical functions (those concerned with prediction accuracy, significance levels)
"""
import numpy as np

def _svd(M,**kwargs):
    try:
        U,S,Vh = np.linalg.svd(M,**kwargs)
    except np.linalg.LinAlgError, e:
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

def _is_numeric(obj):
	attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
	return all(hasattr(obj, attr) for attr in attrs)

def split_data(n_splits,n_tps,is_contiguous=True):
	"""Splits data of length "n_tps" up in to "n_splits" cross-validation chunks
	
	Parameters
	----------
	n_splits : scalar
		number of splits
	n_tps : scalar
		number of time points to subdivide
	is_contiguous : bool
		Governs whether chunks are contiguous
	
	Returns a list ("splits") of (trnIdx,valIdx) pairs of indices
	(Each split is n_tps/n_splits long)
	"""
	splits = []
	n_per_split = n_tps/n_splits
	if is_contiguous:
		indices = np.arange(n_tps)
	else:
		indices = np.random.permutation(n_tps)
	for ii in range(n_splits):
		vIdx = indices[ii*n_per_split:(ii+1)*n_per_split]
		t1 = indices[0:ii*n_per_split]
		t2 = indices[(ii+1)*n_per_split:]
		tIdx = np.concatenate((t1,t2))
		splits.append((tIdx,vIdx))
	return splits

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

def avg_wts(w,nlags=3,skipfirst=True):
    """Average weights for FIR lags. 

    Averages `nlags` blocks of the vector/matrix of weights together. For example,
    a list of 8 weights with nlags=2 would be averaged across the columns of the 
    following matrix (numbers are 1-based indices into the vector of weights):
    1 2 3 4
    5 6 7 8

    for 1D arrays (vectors), assumes 1st dimension is weight dimension;
    for 2D arrays, assumes 2nd dimension is weight dimension.
    """
    nd = np.ndim(w)
    ii = 1 if skipfirst else 0
    if nd==1:
        w = np.mean(w[ii:].reshape(nlags,-1),axis=0)
    elif nd==2:
        sz = w.shape
        w = np.mean(w[:,ii:].reshape(sz[0],nlags,-1),axis=1)
    else:
        raise Exception("I just can't handle it, captain!")
    return w

def best_wts(Mod,roi,n=0,var='cc_norm',avg=True):
    best_cc = sorted(Mod[roi][var],reverse=True)[n]
    idx = Mod[roi][var].tolist().index(best_cc)
    wts = Mod[roi].weights[idx,:]
    if avg:
        wts = su.avg_wts(wts)
    return idx,best_cc,wts

def norm_wts(wts):
    wts/=np.max(np.abs(wts))
    return wts/2.+.5

def progressdot(ii,s1,s2,t):
	"""Display progress through a loop
	
	Keeps tabs on progress through a loop with a display of dots at the
	command line.
	
	Parameters
	----------
	ii : index variable to track 
	s1 : dot every (s1) (prints dot each (s1) ticks through loop)
	s2 : new line every (s2) (prints time and new line each (s2) ticks through loop)
	t : end (max of loop)
	"""
	import time,sys
	def tStr(sec):
		if sec<60:
			ss = '%.1f sec'%sec
		elif sec<3600:
			ss = '%.1f min'%(sec/60.)
		else:
			ss = '%.1f hour'%(sec/3600.)
		return ss
	try:
		#Will throw exception if not set
		progressdot.t0
		# stuff to do each time the function is called:
		# (+1 accounts for zero-first indexing in python)
		if (ii+1)%s1==0:
			sys.stdout.write('.')
		if (ii+1)%s2==0 or ii==t:
			sys.stdout.write('%d/%d done. (%s)\n'%(ii+1,t,tStr(time.time()-progressdot.t0)))
		time.sleep(.00001)
	except AttributeError:
		progressdot.t0 = time.time() # initialize timing variable
		# stuff to call the first time the function is called 
		progressdot(ii,s1,s2,t) #recursive call. Will not throw exception nowpersistent t0

def unwrap(dat,order,n=None):
	"""Unwrap data sequence

	Most often used for validation stimulus sequences, composed 
	of blocks of length `n` in order `ord`

	Parameters
	----------
	dat : array
		First dimension should be time (or dimension to be unwrapped)
	order : array-like, index
		Index for ordering blocks of dat
	n : scalar, int
		length of each block
	"""
	n_values = len(np.unique(order))
	if dat.shape[0]%n_values:
		# Can't handle unequal blocks yet. 
		raise Exception('Data length does not divide evenly into block length!')
	if n is None:
		n = dat.shape[0]/n_values
	
	n_blocks = dat.shape[0]/n
	dat = [dat[ii*n:(ii+1)*n] for ii in range(n_blocks)]
	dat = np.vstack([dat[ii-1] for ii in order])
	return dat

def wrap(dat,order,collapse_fun=np.mean):
	"""Re-wrap data sequence.

	Collapses across repeats. If `collapse_fun` is None, returns 3D array w/ 
	repeats along 3rd axis.
	"""
	n_blocks = len(order)
	if dat.shape[0]%n_blocks:
		raise Exception('Data length does not divide evenly into block length!')
	n = dat.shape[0]/n_blocks
	dat = [dat[ii*n:(ii+1)*n] for ii in range(n_blocks)]
	dd = []
	for value in np.unique(order):
		dd.append(np.dstack([dat[ii] for ii in range(n_blocks) if order[ii]==value]))
	dat = np.vstack(dd)
	if collapse_fun is None:
		return dat
	else:
		return collapse_fun(dat,axis=2)

def contig_partition(n_datapts,n_splits):
    """Evenly partition n_datapts into n_splits

    Each pair is a partition of X, where validation is an iterable
    of length n_datapts/n_splits. 
    """
    nn = np.cast['int32'](np.ceil(np.linspace(0,n_datapts,n_splits+1)))
    val = [np.arange(st,fn) for st,fn in zip(nn[:-1],nn[1:])]
    trn = [np.array([x for x in range(n_datapts) if not x in v]) for v in val]
    return trn,val

def as_list(x):
    """Assure that a variable that might either be a scalar or a list is a list."""
    if x is None:
        return []
    if not isinstance(x,(list,tuple)):
        x = [x]
    return x

def SplitPPseq(ppseq):
	warnings.warn('Deprecated! Use split_ppseq instead!')
	return split_ppseq(ppseq)

def split_ppseq(ppSeq):
	"""Takes a list of preprocessing steps & (potentially multiple) arguments and 
	converts them to a multiple lists
	"""
	argNums = [a for i,a in enumerate(ppSeq) if i%2]
	ppSteps = [a for i,a in enumerate(ppSeq) if not i%2]
	aSeq = list(itertools.product(*argNums))
	ppSeq = []
	for a in aSeq:
		S = []
		for q in zip(ppSteps,[[aS] for aS in a]):
			S+=q
		ppSeq.append(S)

	return ppSeq

def mat_normalize(data,mode='column'):
	'''
	Normalizes (rescales) a vector or matrix such that the values span the
	range from (0 to 1), or (-1 to 1), or whatever, depending on mode).
	
	Parameters
	----------
	data : array-like
		vector or matrix
	mode : string for normalization mode. 
		can be (caps don't matter): 
		'Column' - [default] normalizes max / min values by columns
		'WholeMatrix' - normalizes max value in whole matrix to 1, min to 0
		'WholeMatrixZeroMean' - normalizes max value of matrix to 1, min to 0,
			and then shifts whole matrix so the mean is zero (thus max / min 
			values are no longer 1 / 0, but the range is 1)
		'WholeMatrixRespectZero' - normalizes EITHER the max or min value to 1 or
			-1, but keeps zero wherever it is. 

	'''
	# FIX VECTOR INPUT IF NECESSARY
	#if data.shape[1] > data.shape[0] and min(data.shape)==1:
	#	data = data.T
	if data.dtype in [np.dtype('int64'),np.dtype('int32'),np.dtype('int8'),np.dtype('int16')]:
		# Change integer to float
		data = np.cast['float32'](data)
	mode = mode.lower()
	if mode=='column':
		# Set minimum to zero
		C = data-np.nanmin(data,axis=0)
		Output = C / np.nanmax(C,axis=0)
	elif mode=='wholematrix':
		Output = (data-np.nanmin(data))/(np.nanmax(data)-np.nanmin(data))
	elif mode=='wholematrixzeromean':
		Output = (data-nanmean(data))/(np.nanmax(data-np.nanmin(data)))
	elif mode=='wholematrixrespectzero':
		MaxVal = np.max(np.abs(data))
		Output = data / MaxVal
	else:
		raise Exception('Unknown mode!')
	return Output
	'''
	Old matlab notes on mlNormalize... will translate later
	# NOTE: To reverse (from CCW starting at bottom to CW starting at bottom):
	figure(1);
	subplot(121);
	Pos = mlCirclePos(10,12,0,0);
	ylim([-12 12]); xlim([-12 12]);
	hold on; for i =1:12; text(Pos(i,1),Pos(i,2),num2str(i)); ; hold off
	title('Created from: "Pos = mlCirclePos(10,12,0,0)"')
	subplot(122);
	Pos = mlCirclePos(10,12,12,12,'TopCCW');
	ylim([0 24]); xlim([0 24]);
	hold on; for i =1:12; text(Pos(i,1),Pos(i,2),num2str(i)); ; hold off
	title('Created from: "Pos = mlCirclePos(10,12,12,12,''TopCCW'')"')


	figure(2);
	Pos = mlCirclePos(10,12,0,0,'TopCW');
	ylim([-12 12]); xlim([-12 12]);
	hold on; for i =1:12; text(Pos(i,1),Pos(i,2),num2str(i)); ; hold off
	title('Created from: "Pos = mlCirclePos(10,12,0,0,''TopCW'')"')

	figure(3);
	Pos = mlCirclePos(10,12,0,0,'BotCW');

	ylim([-12 12]); xlim([-12 12]);
	hold on; for i =1:12; text(Pos(i,1),Pos(i,2),num2str(i)); ; hold off
	title('Created from: "Pos = mlCirclePos(10,12,0,0,''BotCW'');"')

	'''
'''
Image Alpha demo:
from PIL import Image
bottom = Image.open("a.png")
top = Image.open("b.png")

r, g, b, a = top.split()
top = Image.merge("RGB", (r, g, b))
mask = Image.merge("L", (a,))
bottom.paste(top, (0, 0), mask)
bottom.save("over.png")
'''