
# coding: utf-8

#####
#data masking code: loads mask and data return the masked data
####

# In[2]:

import matplotlib.pyplot as plt
import numpy as np


# In[3]:

# get_ipython().magic(u'matplotlib inline')


# ## load data

# In[7]:

full_data = np.load("../data/combined_runs.npy")


# In[10]:

full_data.shape


# ## load mask

# In[5]:

mask50k = np.load("../brain_mask/sm_mask55468.npy")


# In[ ]:

mask50k_3d = mask50k.reshape((full_data.shape[0],full_data.shape[1],full_data.shape[2]))


# In[22]:

mask17k = np.load("../brain_mask/sm_mask17887.npy")
mask17k_3d = mask17k.reshape((full_data.shape[0],full_data.shape[1],full_data.shape[2]))


# In[23]:

mask9k = np.load("../brain_mask/sm_mask9051.npy")
mask9k_3d = mask9k.reshape((full_data.shape[0],full_data.shape[1],full_data.shape[2]))


# ## masking

# In[24]:

masked_data_50k = full_data[mask50k_3d,:]
np.save("../data/masked_data_50k.npy",masked_data_50k)



# In[28]:

masked_data_17k = full_data[mask17k_3d,:]
np.save("../data/masked_data_17k.npy",masked_data_17k)


# In[29]:

masked_data_9k = full_data[mask9k_3d,:]
np.save("../data/masked_data_9k.npy",masked_data_9k)


