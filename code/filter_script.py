
""" The following script will use the 3mm Gaussian smoothed data 
(see dataprep_script.py) and take a only a subset of the voxels on this smooted 
data. Specifically it will, 

* Mask voxels based on the FFT and variance threshhold filtering (see brain_mask)
  folder 
* Save a subset of the smooth data consisting of only 
    * 50k voxels 
    * 17k voxels 
    * 9k voxels
* Plot and save figures of the new masked and smoothed dataset 

""" 

#Import Standard Libraries
from __future__ import print_function, division
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

#Local Modules 
import utils.data_loading as dl
import utils.scenes as sc 

files = []
file_root = '../data/smoothed_run_' 

for index in range(8):
    files.append(file_root + str(index) + '.npy')

all_smoothed_data = []
for fl in files:
    #Smoothed data time corrected so don't drop any volumes
    all_smoothed_data.append(np.load(fl)) 

#Masked voxel indcs obtained from FFT and variance threshhold filtering 
mask50k = np.load("../brain_mask/sm_mask55468.npy")

full_data = sc.combine_run_arrays(all_smoothed_data)
all_smoothed_data = None #Save memory 

#Smoothed data for 50k voxels 
mask50k_3d = mask50k.reshape((full_data.shape[0],full_data.shape[1],full_data.shape[2]))

#Smoothed data for 17k voxels 
mask17k = np.load("../brain_mask/sm_mask17887.npy")
mask17k_3d = mask17k.reshape((full_data.shape[0],full_data.shape[1],full_data.shape[2]))

#Smoothed data for 9k voxels 
mask9k = np.load("../brain_mask/sm_mask9051.npy")
mask9k_3d = mask9k.reshape((full_data.shape[0],full_data.shape[1],full_data.shape[2]))

#Save data in the data folder 
masked_data_50k = full_data[mask50k_3d,:]
np.save("../data/masked_data_50k.npy",masked_data_50k)

masked_data_17k = full_data[mask17k_3d,:]
np.save("../data/masked_data_17k.npy",masked_data_17k)

masked_data_9k = full_data[mask9k_3d,:]
np.save("../data/masked_data_9k.npy",masked_data_9k)

####### Visualize new mask #######

#See how brain image at time=1500, z=20 looks 
img_volume = full_data[:,:,:,1500]

#All voxels 
plt.imshow(img_volume[:,:,30])
plt.colorbar()
plt.title('Smoothed: Time = 1500, z = 30')
plt.savefig('../figure/smoothed3mm.jpeg')
plt.close()

#50k mask plot
brain = img_volume.copy().astype('float64')
brain[~mask50k_3d] = np.multiply(brain[~mask50k_3d],0.3).astype('float64')
plt.imshow(brain[:,:,30])
plt.title('Smoothed 50k Voxel Subset: Time = 1500, z = 30')
plt.colorbar()
plt.savefig('../figure/smoothed3mm_50k.jpeg')
plt.close()

#17k mask plot
brain = img_volume.copy().astype('float64')
brain[~mask17k_3d] = np.multiply(brain[~mask17k_3d],0.3).astype('float64')
plt.imshow(brain[:,:,30])
plt.title('Smoothed 17k Voxel Subset: Time = 1500, z = 30')
plt.colorbar()
plt.savefig('../figure/smoothed3mm_17k.jpeg')
plt.close()

#9k mask plot 
brain = img_volume.copy().astype('float64')
brain[~mask9k_3d] = np.multiply(brain[~mask9k_3d],0.3).astype('float64')
plt.imshow(brain[:,:,30])
plt.title('Smoothed 9k Voxel Subset: Time = 1500, z = 30')
plt.colorbar()
plt.savefig('../figure/smoothed3mm_9k.jpeg')
plt.close()
