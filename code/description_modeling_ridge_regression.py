
# coding: utf-8

###
# Code normalize the data and does ridge regression on BOLD response
# output figures of normalized data and regression performance
###

# ## Imports

# In[1]:

from utils import vmt_utils
from utils import regression
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing 
import copy
import time
import os
import nibabel as nib
# import h5py
import cPickle


# In[5]:

data = np.load('../data/filtered_data.npy')


# In[7]:

data_normalized = stats.zscore(data,axis=1,ddof=1)


# In[64]:

plt.plot(data_normalized[16188,:])
plt.xlabel('TRs')
plt.ylabel('normalized voxel repsonse')
plt.savefig('../figure/normalized_data.jpg')


# In[10]:

vvar = np.var(data,axis=1)


# In[11]:

plt.plot(vvar)
plt.ylabel('variance')
plt.xlabel('voxels')
plt.savefig('../figure/data_variance_normalization.jpg')


# ## Get feature space

# In[12]:

# Get stimulus feature spaces
stim_fs_fpath = '../description_pp/design_matrix_1.npy'
stim_fs_file = np.load(stim_fs_fpath)
stim_fs_file = stim_fs_file[:3543,:]


# In[13]:

stim_fs_file.shape


# ## Separate Trainig and testing data

# In[30]:

e, v = 3000, 543


# In[31]:

stim_fs_est = stim_fs_file[:e,:]
stim_fs_val = stim_fs_file[:v,:]


# In[32]:

print('Estimation feature space shape: %s'%repr(stim_fs_est.shape))
print('Validation feature space shape: %s'%repr(stim_fs_val.shape))


# In[33]:

data_est = data_normalized[...,:e]
data_val = data_normalized[...,:v]


# In[34]:

data_est_masked = data_est.T
data_val_masked = data_val.T
# Show size of masked data
print('Size of masked estimation data is %s'%repr(data_est_masked.shape))
print('Size of masked validation data is %s'%repr(data_val_masked.shape))


# ## Set up feature space matrix to do regression

# In[35]:

# Create lagged stimulus matrix
efs = vmt_utils.add_lags(stim_fs_est,[2,3,4])
vfs = vmt_utils.add_lags(stim_fs_val,[2,3,4])
print efs.shape
# Add column of ones
efs = vmt_utils.add_constant(efs,is_first=True)
vfs = vmt_utils.add_constant(vfs,is_first=True)
print efs.shape


# # Subset data and stimuli

# In[36]:

ts = 3000 #training subset
es = 543 #estimation subset


# In[37]:

efs = efs[:ts,:]
vfs = vfs[:es,:]
print efs.shape
print vfs.shape


# In[38]:

data_est_masked = data_est_masked[:ts,:]
data_val_masked = data_val_masked[:es,:]
print data_est_masked.shape
print data_val_masked.shape


# ## Run regression

# In[39]:

reload(vmt_utils)


# In[40]:

alpha = np.logspace(0,6,10)
alpha


# In[41]:

# Run regression 
n_splits = 10 # number of subdivisions of validation data for cross validation of ridge parameter (alpha)
n_resamps = 10 # number of times to compute regression & prediction within training data (can be <= n_splits) #default10
chunk_sz = 10000 # number of voxels to fit at once. Memory-saving.
pthr = 0.005 # Ridge parameter is chosen based on how many voxels are predicted above a correlation threshold 
             # for each alpha value (technically it's slightly more complicated than that, see the code). 
             # This p value sets that correlation threshold.
t0 = time.time()

out = regression.ridge_cv(efs,data_est_masked,val_fs=vfs,val_data=data_val_masked,alphas=alpha,n_resamps=n_resamps,
                              n_splits=n_splits,chunk_sz=chunk_sz,pthr=pthr,is_verbose=True)

t1 = time.time()
print("Elapsed time is: %d min, %d sec"%((t1-t0)/60,(t1-t0)%60))


# ## Make sure estimation procedure chose a reasonable $\alpha$ value
# There should be a somewhat obvious maximum in the curve plotted below

# In[49]:

# # Plot number of voxels with significant prediction accuracy within the 
# # estimation data for each alpha value
# na = len(out['n_sig_vox_byalpha'])
# plt.plot(range(na),out['n_sig_vox_byalpha'],'ko-',lw=2)
# # plt.xticks(range(na),vmt.regression.DEFAULT_ALPHAS,rotation=45)
# plt.xticks(range(na),alpha,rotation=45)
# plt.xlabel('Regularization parameter')
# _ = plt.ylabel('Number of voxels\naccurately predicted')


# ## Display prediction accuracy results on the cortical surface

# In[50]:

out['cc'].shape


# In[51]:

cc = out['cc']
weights = out['weights']


# In[56]:

plt.hist(np.nan_to_num(cc),21)
plt.ylabel('number of voxels')
plt.xlabel('correlation coefficients')
plt.title('correlation coefficient histogram')
plt.savefig('../figure/correlation_coefficient_histogram.jpg')


# In[57]:

out['weights'].shape


# In[58]:

voxel_idx = np.argsort(cc)[::-1][:100]
voxel_idx


# In[69]:

idxx = [13,1780,1014,15500] # hand picked voxels
plt.figure(figsize=(5,10))
for n in range(4):
    i = idxx[n]
    pred_activity = vfs.dot(weights[:,i])
    plt.subplot(4,n//4+1,n%4+1)
    l1, = plt.plot(data_val_masked[:400,i],'b')
    l2, = plt.plot(pred_activity[:400,],'r')
    plt.ylabel('BOLD response')
    plt.xlabel('TRs')
    plt.title('well predict voxel %d\n'%i)
    plt.legend([l1, l2], ['data','prediction'],loc=3)
    
plt.tight_layout()
plt.savefig('../figure/ridge_prediction_results.jpg')


# In[76]:

sum(cc>0.7)


# In[ ]:



