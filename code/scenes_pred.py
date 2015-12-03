
""" The following script will analyze the scenes data. Specifically, it will:

* Try to find patterns between neural responses and scenes   
* Use kmeans and KNN to link these together
* Predict scenes based on BOLD activity 
* Test the performance of the classification through cross-validation  

""" 
#Import Standard Libraries
from __future__ import print_function, division
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

#Local Modules 
import utils.data_loading as dl
import utils.save_files as sv
import utils.scenes as sn

#Clustering Libraries 
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.cluster import KMeans
                                            
#All file strings corresponding to BOLD data for subject 4 
files = ['../data/task001_run001.bold_dico.nii', '../data/task001_run002.bold_dico.nii', 
         '../data/task001_run003.bold_dico.nii', '../data/task001_run004.bold_dico.nii', 
         '../data/task001_run005.bold_dico.nii', '../data/task001_run006.bold_dico.nii',
         '../data/task001_run007.bold_dico.nii', '../data/task001_run008.bold_dico.nii']

#Use only 6810 of the voxels (choose most variance-stable voxels for further analysis)
SUBSET_VOXELS_PATH = '../brain_mask/mask6810.npy'
mask = np.load(SUBSET_VOXELS_PATH) #mask 
VOXEL_INDCS = np.where(mask == True)[0]

unraveled = []
for one_d_pos in VOXEL_INDCS:
    three_d_pos = np.unravel_index(one_d_pos, (160, 160, 36))
    unraveled.append(three_d_pos)
unraveled = np.array(unraveled)

first_comp = unraveled[:,0]
sec_comp = unraveled[:,1]
third_comp = unraveled[:,2]

#Load in data 
all_data = []
for index, filename in enumerate(files):
    new_data = dl.load_data(filename) #load_data function drops first 4 for us
    new_data = new_data[first_comp, sec_comp, third_comp, :] #vox by time array 2d
    num_vols = new_data.shape[-1]
    if index != 0 and index != 7:
        new_num_vols = num_vols - 4   
        new_data = new_data[:,:new_num_vols] #Drop last 4 volumes for middle runs    
    all_data.append(new_data)

scenes_path = '../data/scene_times_nums.csv'
scenes = pd.read_csv(scenes_path, header = None) 
scenes = scenes.values #Now just a numpy array

combined_runs = np.concatenate(all_data, axis = 1) 
combined_runs = combined_runs[:,9:] #First 17 seconds are credits/no scene id so drop
all_data = [] #Dont need to store this anymore

TR = 2
NUM_VOLUMES = combined_runs.shape[-1] #3459 
ONSET_TIMES = scenes[:,0] 
ONSET_TIMES_NORMED = ONSET_TIMES - 17 #First recorded scene occurs at t = 17 sec 
DURATION = scenes[:,1] 
LABELS = scenes[:,3]
SCAN_TIMES =  np.arange(start=0, stop=2*NUM_VOLUMES, step=2)

#Creates a list that tells us scene id at given scan time 
factor_grid = []
for scan_time in SCAN_TIMES:
    index_list = np.where(ONSET_TIMES_NORMED < scan_time)[0]
    if scan_time == 0:
        label_index = 0
    else:
        label_index = index_list[-1] 
    factor_id = LABELS[label_index]
    factor_grid.append(factor_id)

factor_grid = np.array(factor_grid) #Convert to np array for future analysis

#Grouped Factors Ids 
GUMP_SCENES_IDS = [38, 40, 41, 42] #factor ids of Gump scenes
MILITARY_IDS = [52, 62, 77, 78, 80, 81, 82, 83]
SCHOOL = [22,43, 67, 61, 69]
SAVANNA = [66]
POLITICAL = [86, 2, 87, 84]
OUTSIDE = [27, 73, 58, 53, 59]

############K-means Analysis ##################################

#Comparison between Military and Gump Scenes 
all_ids_1 = GUMP_SCENES_IDS + MILITARY_IDS 
samp_1, miss_1 = sn.gen_sample_by_factors(all_ids_1, factor_grid, True, prop = .9)
training1 = sn.get_training_samples(samp_1)
train_labs1, train_times1 = sn.make_label_by_time(training1)
milt_subarr = combined_runs[:,train_times1]

kmeans = KMeans(n_clusters=2, n_init=10)
pred1 = kmeans.fit_predict(milt_subarr.T)

#Check accuracy 
#Make a vector that is 1 for Gump Scenes and 0 otherwise 
on_off_1 = sn.on_off_course(GUMP_SCENES_IDS, train_labs1)
num = sn.analyze_performance(pred1, on_off_1)
accuracy1 = max(num, 1 - num) 

#Comparison between 




