
""" The following script will analyze the scenes data. Specifically, it will:

* Try to find patterns between neural responses and scenes   
* Use SVM and KNN to link these together
* Predict scenes based on BOLD activity 
  
"""  
#Import Standard Libraries
from __future__ import print_function, division
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import itertools
from pylab import *

#Local Modules 
import utils.data_loading as dl
import utils.save_files as sv
import utils.scenes as sn

#Clustering Libraries 
from sklearn import preprocessing as pp 
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
                                            
#Load in filtered data and normalize 
masked_path = "../data/filtered_data.npy"
combined_runs = pp.normalize(np.transpose(np.load("../data/filtered_data.npy")))

#Too many predictors (55k) - filter to around 1500 predictors
xvar = np.var(combined_runs, axis=0)
varmask = np.where(xvar > .0000000015)[0]
combined_runs = combined_runs.T[varmask] #1584 voxels 

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

############ SVM and KNN Analysis #################################

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

subarr1_train = combined_runs[:,train1_times].T #rows correspond to images, colums to voxels 
subarr1_test = combined_runs[:,test1_times].T #data we feed into our classifier 

clf = svm.SVC(C=100, kernel='linear') #Paramters obtained through cross-validation
clf.fit(subarr1_train, on_off1_train)
pred_svm1 = clf.predict(subarr1_test)
accuracy_score(on_off1_test, pred_svm1) #52%

knn = KNeighborsClassifier()
knn.fit(subarr1_train, on_off1_train)
pred_knn1 = knn.predict(subarr1_test)
accuracy_score(on_off1_test, pred_knn1) #69%

#Compare more scenes 
all_ids_2 = GUMP_SCENES_IDS + SCHOOL + MILITARY_IDS + SAVANNA + POLITICAL + OUTSIDE + DEATH + CHURCH
sample2, missing_facts2 = sn.gen_sample_by_factors(all_ids_2, factor_grid, True, prop=.9)
train_samp2 = sn.get_training_samples(sample2)
test_samp2 = sn.get_tst_samples(sample2)

train2_labs, train2_times = sn.make_label_by_time(train_samp2)
test2_labs, test2_times = sn.make_label_by_time(test_samp2)

#Set up ids for each category 
labels2_train = []
for val in train2_labs:
    if val in GUMP_SCENES_IDS:
        labels2_train.append(0)
    elif val in SCHOOL:
        labels2_train.append(1)
    elif val in MILITARY_IDS:
        labels2_train.append(2)
    elif val in SAVANNA:
        labels2_train.append(3)
    elif val in POLITICAL:
        labels2_train.append(4)
    elif val in OUTSIDE:
        labels2_train.append(5)
    elif val in DEATH:
        labels2_train.append(6)
    else:
        labels2_train.append(7)

labels2_train = np.array(labels2_train)

labels2_test = []
for val in test2_labs:
    if val in GUMP_SCENES_IDS:
        labels2_test.append(0)
    elif val in SCHOOL:
        labels2_test.append(1)
    elif val in MILITARY_IDS:
        labels2_test.append(2)
    elif val in SAVANNA:
        labels2_test.append(3)
    elif val in POLITICAL:
        labels2_test.append(4)
    elif val in OUTSIDE:
        labels2_test.append(5)
    elif val in DEATH:
        labels2_test.append(6)
    else:
        labels2_test.append(7)

labels2_test = np.array(labels2_test)

subarr2_train = combined_runs[:,train2_times].T 
subarr2_test = combined_runs[:,test2_times].T

clf = svm.SVC(C=100, kernel='linear') #Paramters obtained through cross-validation 
clf.fit(subarr2_train, labels2_train)
pred_svm2 = clf.predict(subarr2_test)

accuracy_score(labels2_test, pred_svm2) #27.7%

knn = KNeighborsClassifier()
knn.fit(subarr2_train, labels2_train)
pred_knn2 = knn.predict(subarr2_test)

accuracy_score(labels2_test, pred_knn2) #34% 

#Knn looks better - let's see how it performs by cateogry 

#Check performance over the 6 categories 
gump_indcs = np.where(labels2_test == 0)[0]
school_inds = np.where(labels2_test == 1)[0]
milit_incs = np.where(labels2_test == 2)[0]
savan_indcs = np.where(labels2_test == 3)[0]
political_indcs = np.where(labels2_test == 4)[0]
outside_indcs = np.where(labels2_test == 5)[0]
death_indcs = np.where(labels2_test == 6)[0]
church_inds = np.where(labels2_test == 7)[0]

by_cat = [gump_indcs, school_inds, milit_incs, savan_indcs, political_indcs, 
          outside_indcs, death_indcs, church_inds]

perform_by_cat = []
actual_count = []
pred_count = []

for scence_ind in by_cat:
    acc = accuracy_score(labels2_test[scence_ind], pred_knn2[scence_ind])
    weight = scence_ind.shape[0]
    perform_by_cat.append(acc)
    actual_count.append(weight)
     
#Plot this
actual_count = np.array(actual_count)
relative_weights = actual_count / sum(actual_count)

#create labels for pie chart 
categories = ['gump', 'school', 'military', 'savanna', 'political',
              'outside', 'death', 'church']

categories_per = []
for index, name in enumerate(categories):
    name2 = name + ': ' + '' + str(round(perform_by_cat[index], 3) * 100) + '%'
    categories_per.append(name2)

pie(relative_weights, labels=categories_per,autopct='%1.1f%%')
plt.title('Category Weight and Performance by Category')
plt.savefig('../figure/scenes_pie_chart.png')
plt.close()
