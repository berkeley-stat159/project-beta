
""" The following script will analyze the scenes data. Specifically, it will:

* Use PCA to reduce data/noise for fitting 
* Try to find patterns between neural responses and scenes   
* Use regression, SVM, and KNN to link these together
* Predict scenes based on BOLD activity 
* Test the performance of the classification through cross-validation  

""" 

#ADD PATHS
 

#Import standard libraries
from __future__ import print_function, division
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import data_loading_script as dl
import plotting_fmri as plt_fmri
import save_files as sv
import numpy.linalg as npl

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

#DELETE THESE TWO FUNCTIONS ALREADY IN DATA_LOADING (b/c paths not right)
def load_data(filename):
    """ Return fMRI data corresponding to the given filename and prints
        shape of the data array
    ----------
    filename : string 
        This string should be a path to given file. The file must be
        .nii 
    Returns
    -------
    data : numpy array
        An array consisting of the data. 
    """
    img = nib.load(filename)
    data = img.get_data()
    data = data[:,:,:,4:]
    print(data.shape)
    return data

def voxel_by_time(data):
    n_voxels = np.prod(data.shape[:-1])
    return np.reshape(data, (n_voxels, data.shape[-1]))


#FIX: PUT THESE FUNCTIONS SOMEWHERE IN UTILS
###################################################################
ALL_IDS = list(range(1, 91))

def get_factor_ids(times, factor_grid):
    """ Returns factor ids that occured at 'times'   
    
    Parameters
    ----------
    times : list
    Consists of scan times (multiples of TR = 2 sec)
    factor_grid: numpy array 
        At index i, this corresponds to the scene that occured at scan time (sec) i   
    Returns
    -------
    numpy array 
        An array consisting of factor ids at correponding time 
    """
    return factor_grid[times]

def on_off_course(on_fact_ids, factor_grid):
    """ Returns a list of 0's and 1's at each scan time in 'SCAN_TIMES' list. 
        A 0 entry indicates the scene id at the corresponding scan time entry 
        was not contained in 'on_fact_ids'. A 1 indicates the exact opposite.    
    
    Parameters
    ----------
    on_fact_ids : list
        All scene/factor ids to set to 1 in the output list. (see above)
    factor_grid: numpy array 
        At index i, this corresponds to the scene that occured at scan time (sec) i   
    Returns
    -------
    other_ids : numpy array 
        An array of 0's and 1's
    """
    on_off_times = []
    for fact_id in factor_grid:
        if fact_id in on_fact_ids:
            on_off_times.append(1)
        else:
            on_off_times.append(0)
    return np.array(on_off_times)

def multiple_factors_course(on_fact_ids, factor_grid):
    """ Returns a list of 0's and ids in 'on_fact_ids'. 
        A 0 entry indicates the scene id at the corresponding scan time entry 
        was not contained in 'on_fact_ids'. Otherwise a value i indicates
        that scene/factor id i in 'on_fact_ids' occured at this scan time.    
    
    Parameters
    ----------
    on_fact_ids : list
        All scene/factor ids to set in the output list. (see above)
    factor_grid: numpy array 
        At index i, this corresponds to the scene that occured at scan time (sec)         
    Returns
    -------
    other_ids : numpy array
        A list consisting of 0's and ids in 'on_fact_ids'
    """
    on_off_times = []
    for fact_id in factor_grid:
        if fact_id in on_fact_ids:
            on_off_times.append(fact_id)
        else:
            on_off_times.append(0)
    return np.array(on_off_times)

def combine_run_arrays(run_array_lst):
    """ Returns a combined 4d array by concatinating based on time (axis = 3)
    ----------
    run_array_lst : array of 4d numpy arrays 

    Returns
    -------
    4d numpy array 
    """
    return np.concatenate(run_array_lst, axis = 3)

def all_factors_indcs(factor_lst, factor_grid, min_time=0, max_time=3459):
    """ Returns a dictionary with numpy arrays as values. Each numpy array 
    consists of the indcs at which the corresponding factor id occured in 'factor_grid'
    within 'min_time' and 'max_time'    
    
    Parameters
    ----------
    factor_lst : list  
        Scene/factor ids  
    factor_grid: numpy array 
        At index i, this corresponds to the scene that occured at scan time (sec) 
    min_time: int
        Minimum index (scan time) to look at in 'factor_grid' 
    max_time: int   
        Maximum index (scan time) to look at in 'factor_grid'     
    Returns
    -------
    all_factors : dictionary of numpy arrays
        Each value consists of idcs for corresponding factor id (see above)  
    """
    all_factors = {}
    constrained_grid = factor_grid[min_time:max_time] 
    for factor in factor_lst:
        factor_indcs = np.where(constrained_grid == factor)[0] #Get desired list from tuple 
        all_factors[factor] = factor_indcs
    return all_factors


def gen_sample_by_factors(factor_lst, factor_grid, randomize, prop=.5, min_time=0, max_time=3459):
    """ Returns a dictionary and a list. The key of the dictionary is the factor_id
    and the value is a tuple. The first element of the tuple is numpy array of
    training indcs and the second element is numpy array of testing indcs. The
    input 'prop' allocates what proportion of all indcs the training and testing
    samples recieve. If 'randomize' is true, the indcs will be randomly assigned 
    to the training and testing arrays (instead of sequentially). The returned
    list 'missing_factors' corresponds to factor ids of the excluded factor ids 
    due to insufficient indcs list sizes. 
   
    Parameters
    ----------
    factor_lst : list 
        Scene/factor ids  
    factor_grid: numpy array 
        At index i, this corresponds to the scene that occured at scan time (sec)
    randomize: boolean 
        'True' randomizes the assigment to training and test sets  
    min_time: int
        Minimum index (scan time) to look at in 'factor_grid' 
    max_time: int   
        Maximum index (scan time) to look at in 'factor_grid'     
    Returns
    -------
    (sample, missing_factors) : (dictionary, list) 
        See above  
    """
    sample = {}
    missing_factors = []
    all_factors = all_factors_indcs(factor_lst, factor_grid, min_time, max_time)
    for factor, factor_indcs in all_factors.iteritems():
        samp_size = len(factor_indcs)
        num_train = np.round(prop * samp_size)
        num_test = samp_size - num_train
        if num_train > 0 and num_test > 0:
            if randomize:
                shuffled = np.random.permutation(factor_indcs)
            else: 
                shuffled = factor_indcs 
            train = shuffled[:num_train]
            test = shuffled[num_train:]
            sample[factor] = (train, test)
        else:
            missing_factors.append(factor)
    return (sample, missing_factors)

def get_training_samples(samples):
    """ Returns only the training indcs in 'samples.' 
    Note: 'samples' is the returned dictionary in the 'gen_sample_by_factors'
    function above 
    
    Parameters
    ----------
    samples : dictionary
        Returned dictionary in 'gen_sample_by_factors'
    Returns
    -------
    training : dictionary
    """
    training = {}
    for factor, sample in samples.iteritems():
        training[factor] = sample[0]
    return training

def get_testing_samples(samples):
    """ Returns only the testing indcs in 'samples.' 
    Note: 'samples' is the returned dictionary in the 'gen_sample_by_factors'
    function above 
    
    Parameters
    ----------
    samples : dictionary
        Returned dictionary in 'gen_sample_by_factors'
    Returns
    -------
    testing : dictionary
    """
    testing = {}
    for factor, sample in samples.iteritems():
        testing[factor] = sample[1]
    return testing

def make_label_by_time(sing_samp):
    """ Return a tuple that consists of labels and their according time. 
    
    Parameters
    ----------
    sing_samp : dictionary
        The key of the dictionary is the factor_id
    and the value is a tuple. The element of the tuple is 
    training indcs or testing indcs. The input 'prop' allocates what 
    proportion of all indcs the training and testing samples recieve. 
    Note: 'samples' is the returned dictionary in the 'gen_sample_by_factors'
    
    Returns
    -------
    labels:
        numpy array that contains the labels
    times:
        numpy array that contains the times
    """
    factors = []
    lengths = []
    times = []
    for factor, samp in sing_samp.iteritems():
        factors.append(factor)
        lengths.append(len(samp))
        times.append(samp)
    times = [time for time_lst in times for time in time_lst] 
    times = np.array(times)
    labels = np.repeat(factors, lengths)
    return (labels, times)

def other_scene_ids(remove_ids):
    """ Return list of ids that do not contain any ids present in remove_ids. 
    
    Parameters
    ----------
    remove_ids : list
        This is a list consisting of all ids to remove from the factor 
        scene ids ALL_IDS (which consist of ids between 1 - 90 inclusive)      
    Returns
    -------
    other_ids : list
        A list of ids that do not contain the any of the ids in remove_ids
    """
    assert max(remove_ids) < 91 and min(remove_ids) > 0
    other_ids = [i for i in ALL_IDS if i not in remove_ids]
    return other_ids

#Regression 

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
####################################################################

#We will first check if there are any noticable differences between scenes
#occuring in 'Gump house/room/etc.' vs. all other scenes 

#In this simple example, we let 1 denote a Gump scene and 0 for all other 
#scenes 


GUMP_SCENES_IDS = [38, 40, 41, 42] #factor ids of Gump scenes
other_scenes = other_scene_ids(GUMP_SCENES_IDS)



#Next we will look at political scenes vs. non-political scenes 

#Here we investigate outdoors vs. indoor scenes 

#The following compares the 6 major scenes in the film 





