
""" The following script will analyze the scenes data. Specifically, it will:

* Use PCA to reduce data/noise for fitting 
* Try to find patterns between neural responses and scenes   
* Use regression, SVM, and KNN to link these together
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
import utiles.scenes as scenes

#Clustering Libraries 
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.cluster import KMeans
from scipy.spatial.distance import hamming 
                                            

#All file strings corresponding to BOLD data for subject 4 

files = ['task001_run001.bold_dico.nii', 'task001_run002.bold_dico.nii', 
         'task001_run003.bold_dico.nii', 'task001_run004.bold_dico.nii', 
         'task001_run005.bold_dico.nii', 'task001_run006.bold_dico.nii',
         'task001_run007.bold_dico.nii', 'task001_run008.bold_dico.nii']

#Load in data 
all_data = []
for index, filename in enumerate(files):
    new_data = load_data(filename) #load_data function drops first 4 for us
    num_vols = new_data.shape[-1]
    if index != 0 and index != 7:
        new_num_vols = num_vols - 4   
        new_data = new_data[:,:,:,:new_num_vols] #Drop last 4 volumes for middle runs    
    print(new_data.shape[-1])
    all_data.append(new_data)

scenes_path = '../../data/scene_times_nums.csv' #FIX
scenes = pd.read_csv('scene_times_nums.csv', header = None) 
scenes = scenes.values #Now just numpy array

combined_runs = combine_run_arrays(all_data) 
combined_runs = combined_runs[:,:,:,9:] #First 17 seconds are credits/no scene id so drop

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


#FIX: PUT THESE FUNCTIONS SOMEWHERE IN UTILS

#Clean up data before analysis (choose principal components to use for each scene)

#Check if different scenes result in significatly different principal components 
pca = PCA(n_components=20)



###################################################################
#Set up training and test set for Gump 
GUMP_SCENES_IDS = [38, 40, 41, 42] #factor ids of Gump scenes
other_scenes = other_scene_ids(GUMP_SCENES_IDS)

samp_gump, miss_gump = gen_sample_by_factors(GUMP_SCENES_IDS, factor_grid, True)
training_gump = get_training_samples(samp_gump)
testing_gump = get_testing_samples(samp_gump)

train_labs_gump, train_times_gump = make_label_by_time(training_gump)
test_labs_gump, test_times_gump = make_label_by_time(testing_gump)

#SVM 
gump_subarr = combine_run_arrays[:,:,:,train_times_gump]
vox_by_time_train = voxel_by_time(gump_subarr)
clf = svm.SVC()


####################################################################
#DELETE ALL THIS - JUST CHECKING IF WORKS 

train = combined_runs[:,:,:,447:500] #run 2
train_labels = on_off_course([66], factor_grid) #random factor ids in time interval fix
train_labels = train_labels[447:500]
train_vox_time = voxel_by_time(train)

test = combined_runs[:,:,:,500:600]
true_labels = on_off_course([66], factor_grid)
true_labels = true_labels[500:600]
test_vox_time = voxel_by_time(test)

#Sample KNN example - FIX NEED TO CUT DOWN DIMENSION W/ PCA TO DO BETTER
#Use PCA to find 'good' voxels that seem to predict well - probably need to do
#this by breaking up by the scene and then for each scene group doing PCA 

knn = KNN()

#Kmeans 
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(train_vox_time.T) #Fix should specify labels

#SVM 
clf = svm.SVC()
clf.fit(train_vox_time.T, train_labels)
clf.predict(test_vox_time.T)

