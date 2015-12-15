
""" The following script will apply a 3mm Gaussian filter on all the data spatially
and will save each smoothed run into the data folder as 'smoothed_run_i', where  
0 <= i <= 7 is the index of the run. 
"""

#Import libraries
import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter

import nibabel as nb
import matplotlib.pyplot as plt
import utils.data_loading as dl

#All file strings corresponding to BOLD data for subject 4 
files = ['../data/task001_run001.bold_dico.nii.gz', '../data/task001_run002.bold_dico.nii.gz', 
         '../data/task001_run003.bold_dico.nii.gz', '../data/task001_run004.bold_dico.nii.gz', 
         '../data/task001_run005.bold_dico.nii.gz', '../data/task001_run006.bold_dico.nii.gz',
         '../data/task001_run007.bold_dico.nii.gz', '../data/task001_run008.bold_dico.nii.gz']

all_data = []
for index, filename in enumerate(files):
    new_data = dl.load_data(filename) #load_data function drops first 4 for us
    num_vols = new_data.shape[-1]
    if index != 0 and index != 7:
        new_num_vols = num_vols - 4   
        new_data = new_data[:,:,:,:new_num_vols] #Drop last 4 volumes for middle runs    
    all_data.append(new_data)

#Create an array of all smoothed data  
for index, run in enumerate(all_data):
    num_vols = np.shape(run)[-1]
    run_i_smoothed = []
    for time in range(num_vols):
        smoothed = dl.smooth_gauss(run, 3, time)
        smoothed.shape = (132, 175, 48, 1)
        run_i_smoothed.append(smoothed)
    run_i_smoothed = np.concatenate(run_i_smoothed, axis = 3)
    np.save('../data/smoothed_run_' + str(index), run_i_smoothed) #save in data folder
    print('finished run' + str(index))
    run_i_smoothed = None #Save memory space 
