""" The following script will preprocess the data. Specifically, it will:

* Run diagnostic analysis on FMRI run as in HW2 
* Write oulier indices to text files to use in statistical_analysis_script.py
* Plot samples slices of different runs on the data w/ & w/o outliers  
* Run PCA to check which voxels seem to have the most signal 
* Use linear regression and convolution to model BOLD Signal

* Note: All plots and figures will be saved with the file hierarchy in 
        'save_files.py'  
"""   

""" Import standard libraries """

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import data_loading as dl
import plotting_fmri as plt_fmri
import save_files as sv

""" All file strings corresponding to BOLD data for subject 4 """ 

files = ['task001_run001.bold_dico.nii', 'task001_run002.bold_dico.nii', 
         'task001_run003.bold_dico.nii', 'task001_run004.bold_dico.nii', 
         'task001_run005.bold_dico.nii', 'task001_run006.bold.nii'
         'task001_run007.bold.nii', 'task001_run008.bold.nii']

"""
* Load the images as an image object
* Load all the image data from the images
* Drop the first four volumes, as we know these are outliers
"""

all_data = []
for filename in files:
	new_data = dl.load_data(filename) #load_data function drops first 4 for us
	all_data.append(new_data)

""" 
* Get indices of outlier volumes for each dataset. 
* Write each as its own file and save in 'vol_std_outliers' folder 
* Takes 15 min to run
"""
all_bands_outliers = []
all_sdevs = []
all_iqr_outliers = []
for data in all_data:
	sdev = dl.vol_std(data) 
	all_sdevs.append(sdev)
	outlier, band = dl.iqr_outliers(sdev)
	all_iqr_outliers.append(outlier)
	all_bands_outliers.append(band)

sv.save_all(all_sdevs, fileroot='sdevs', typ='data', folder_root='SDEVS'
	            ext='txt')

sv.save_all(all_iqr_outliers, fileroot='out_iqr', typ = 'data', 
	            folder_root='OUTLIER_IQRs', ext='txt')

sv.save_all(all_bands_outliers,fileroot='band',typ='data',folder_root='IQR_BANDS'
	            ext='txt')

"""  
For each run, we have a plot of:

* The volume standard deviation values;
* The outlier points from the std values, marked on the plot with an 'o'
  marker;
* A horizontal dashed line at the lower IRQ threshold;
* A horizontal dashed line at the higher IRQ threshold

"""
 
for index, sdevs in enumerate(all_sdevs):
	outlier_sdevs = all_iqr_outliers[index]
	outlier_interval = all_bands_outliers[index]
	plt_fmri.plot_sdevs(sdevs, outlier_sdevs, outlier_interval)
	sv.save_plt(fileroot='vol_std_plt', index=index, folder_root='VOL_STD_PLTS',ext='png')
    plt.close()

#Do the same for the RMS 
all_rms = []
all_outliers_rms = []
all_bands_rms = []
for data in all_data:
	rms = dl.vol_rms_diff(data)
	outliers_rms, rms_interval = dn.iqr_outliers(rms)
	all_rms.append(rms)
	all_outliers_rms.append(outliers_rms)
	all_bands_rms.append(rms_interval)

for index, rms in enumerate(all_rms):
	outlier_rms = all_outliers_rms[index]
	outlier_interval = all_bands_rms[index]
	plt_fmri.plot_rms(rms, outlier_rms, outlier_interval)
	sv.save_plt(fileroot='vol_rms_outliers', index=index, folder_root='VOL_RMS_PLTS',ext='png')
    plt.close()
