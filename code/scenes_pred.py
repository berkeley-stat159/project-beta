
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
import itertools

#Local Modules 
import utils.data_loading as dl
import utils.save_files as sv
import utils.scenes as sn

#Clustering Libraries 
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
                                            
#All file strings corresponding to BOLD data for subject 4 
files = ['../data/task001_run001.bold_dico.nii', '../data/task001_run002.bold_dico.nii', 
         '../data/task001_run003.bold_dico.nii', '../data/task001_run004.bold_dico.nii', 
         '../data/task001_run005.bold_dico.nii', '../data/task001_run006.bold_dico.nii',
         '../data/task001_run007.bold_dico.nii', '../data/task001_run008.bold_dico.nii']

#Load in filtered data 
masked_path = "../data/masked_data_9k.npy"
combined_runs = np.load(masked_path)

#Load in scenes data 
scenes_path = '../data/scene_times_nums.csv'
scenes = pd.read_csv(scenes_path, header = None) 
scenes = scenes.values #Now just a numpy array

TR = 2
NUM_VOLUMES = combined_runs.shape[-1] #3543 
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
POLITICAL = [86, 85, 2, 87, 84]
OUTSIDE = [27, 73, 58, 53, 59]
CHURCH = [20]
DEATH = [16, 48]
BARBER = [8]

############ SVM Analysis #################################

#Comparison between Military and Gump Scenes 

#Set up training and testing samples and data 
all_ids_1 = GUMP_SCENES_IDS + MILITARY_IDS 
sample1, missing_facts1 = sn.gen_sample_by_factors(all_ids_1, factor_grid, True, prop=.9)
train_samp1 = sn.get_training_samples(sample1)
test_samp1 = sn.get_tst_samples(sample1)

train1_labs, train1_times = sn.make_label_by_time(train_samp1)
test1_labs, test1_times = sn.make_label_by_time(test_samp1)

on_off1_train = sn.on_off_course(GUMP_SCENES_IDS, train1_labs)
on_off1_test = sn.on_off_course(GUMP_SCENES_IDS, test1_labs)

subarr1_train = combined_runs[:900,train1_times].T #rows correspond to images, colums to voxels 
subarr1_test = combined_runs[:900,test1_times].T #data we feed into our classifier 

clf = svm.SVC(C=100, kernel='linear') 
clf.fit(subarr1_train, on_off1_train)
pred_svm1 = clf.predict(subarr1_test)

knn = KNeighborsClassifier()
knn.fit(subarr1_train, on_off1_train)
pred_knn1 = knn.predict(subarr1_test)

#Compare more scenes 
all_ids_2 = GUMP_SCENES_IDS + SCHOOL + MILITARY_IDS + SAVANNA + POLITICAL + OUTSIDE
sample2, missing_facts2 = sn.gen_sample_by_factors(all_ids_2, factor_grid, True, prop=.9)
train_samp2 = sn.get_training_samples(sample2)
test_samp2 = sn.get_tst_samples(sample2)

train2_labs, train2_times = sn.make_label_by_time(train_samp2)
test2_labs, test2_times = sn.make_label_by_time(test_samp2)

labels2_train = sn.multiple_factors_course(GUMP_SCENES_IDS, train2_labs)
labels2_test = sn.multiple_factors_course(GUMP_SCENES_IDS, test2_labs)

subarr2_train = combined_runs[:900,train2_times].T 
subarr2_test = combined_runs[:900,test2_times].T

clf = svm.SVC(C=100, kernel='linear') #Paramters obtained through cross-validation 
clf.fit(subarr2_train, train2_labs)
pred_svm2 = clf.predict(subarr2_test)

accuracy_score(test2_labs, pred_svm2) #78%

knn = KNeighborsClassifier()
knn.fit(subarr2_train, train2_labs)
pred_knn2 = knn.predict(subarr2_test)

accuracy_score(test2_labs, pred_knn2) #58% 
