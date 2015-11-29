#FIX ALL PATHS SINCE NOT CORRECT
#PROBABLY WANT TO DO THIS ON CLEANED DATA 
#ADD TESTS 

#Import standard libraries
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import data_loading as dl
import plotting_fmri as plt_fmri
import save_files as sv
import numpy.linalg as npl

from __future__ import print_function, division
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.cluster import KMeans
from scipy.spatial.distance import hamming 
                                            

#All file strings corresponding to BOLD data for subject 4 

files = ['task001_run001.bold_dico.nii', 'task001_run002.bold_dico.nii', 
         'task001_run003.bold_dico.nii', 'task001_run004.bold_dico.nii', 
         'task001_run005.bold_dico.nii', 'task001_run006.bold_dico.nii',
         'task001_run007.bold_dico.nii', 'task001_run008.bold_dico.nii']

all_data = []
for index, filename in enumerate(files):
    new_data = dl.load_data(filename) #load_data function drops first 4 for us
    num_vols = new_data.shape[-1]
    if index != 0 and index != 7:
        new_num_vols = num_vols - 4   
        new_data = new_data[:,:,:,:new_num_vols] #Drop last 4 volumes for middle runs    
    print(new_data.shape[-1])
    all_data.append(new_data)

scenes_path = '../../data/scene_times_nums.csv' #FIX
scenes = pd.read_csv('scene_times_nums.csv', header = None) 
scenes = scenes.values #Now just numpy array

combined_runs = combine_run_arrays(all_data) #FIX AND ADD runi
combined_runs = combined_runs[:,:,:,9:] #First 17 seconds are credits/no scene id so drop

TR = 2
NUM_VOLUMES = combined_runs.shape[-1] #3459
ONSET_TIMES = scenes[:,0] 
ONSET_TIMES_NORMED = ONSET_TIMES - 17 #Dropped first 9 volumes which corresponds to 18 sec
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

#Set up training and test set 
train = combined_runs[:,:,:,447:500] #run 2
train_labels = on_off_course([66]) #random factor ids in time interval fix
train_labels = train_labels[447:500]
train_vox_time = voxel_by_time(train)

test = combined_runs[:,:,:,500:600]
true_labels = on_off_course([66])
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


#FIX: PUT THESE FUNCTIONS SOMEWHERE IN UTILS
###################################################################
def on_off_course(on_fact_ids):
    on_off_times = []
    for fact_id in factor_grid:
        if fact_id in on_fact_ids:
            on_off_times.append(1)
        else:
            on_off_times.append(0)
    return on_off_times

def multiple_types_course(on_fact_ids):
    on_off_times = []
    for fact_id in factor_grid:
        if fact_id in on_fact_ids:
            on_off_times.append(fact_id)
        else:
            on_off_times.append(0)
    return on_off_times

def combine_run_arrays(run_array_lst):
    return np.concatenate(run_array_lst, axis = 3)

def gen_train_byID(factor_id):

def train_test_split(vox_by_time, train_times):

def get_index_scene(factor_id, time_window):

def get_scenes(ids):

def other_scene_ids(remove_ids):
    """ Return list of ids that do not contain any ids present in remove_ids. 
    Parameters
    ----------
    remove_ids : list
        This is a list consisting of all ids to remove from the factor 
        scene ids (which consist of ids between 1 - 90 inclusive)      
    Returns
    -------
    other_ids : list
        A list of ids that do not contain the any of the ids in remove_ids
    """
    assert max(remove_ids) < 91 and min(remove_ids) > 0
    all_ids = list(range(1, 91))
    other_ids = [i for i in all_ids if i not in remove_ids]
    return other_ids

def make_scene_design_mat(scenes, times, on_scene_ids):
    synced_times = sync_scene_times(times)

def calc_weights(X, Y):
    return npl.pinv(X).dot(Y)

def get_voxel_weight(vox_by_time, voxel_index, times, on_scene_ids):
    vox_time_course = vox_by_time[voxel_index, times]
    vox_design_mat = make_scene_design_mat(vox_by_time, voxel_index, times, on_scene_ids)
    weights = calc_weights(vox_design_mat, vox_time_course)
    return weights

def get_all_weights(vox_by_time, times, on_scene_ids):
    all_weights = []
    for i in range(vox_by_time.shape[0]):
        weight = get_voxel_weight(vox_by_time, i, times, on_scene_ids)
        all_weights.append(weight)
    all_weights = np.array(all_weights)
    return all_weights

def predict_scene(trained_weights, t_0_activity):
    #ASSERTS HERE FOR DIMENSION?
    design_mat = trained_weights.T
    raw_predicted_scenes = calc_weights(design_mat, t_0_activity)
    largest_index = np.where(raw_predicted_scenes == max(raw_predicted_scenes))
    return largest_index #FIX what if there are ties 

def predict_all_times(trained_weights, test_activities):
    predicted_indces = []
    for activity in test_activities:
        index = predict_scene(trained_weights, activity)
    return predicted_indces

def analyze_performance(predicted_labels, actual_labels): 
    normed_distance = hamming(predicted_labels, actual_labels) #between 0 - 1
    return 1 - normed_distance
    


###################################################################

#We will first check if there are any noticable differences between scenes
#occuring in 'Gump house/room/etc.' vs. all other scenes 

#In this simple example, we let 1 denote a Gump scene and 0 for all other 
#scenes 


GUMP_SCENES_IDS = [38, 40, 41, 42] #factor ids of Gump scenes
other_scenes = other_scene_ids(GUMP_SCENES_IDS)



#Next we will look at political scenes vs. non-political scenes 

#Here we investigate outdoors vs. indoor scenes 

#The following compares the 6 major scenes in the film 





