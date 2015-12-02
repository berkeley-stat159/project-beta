from __future__ import print_function, division
import numpy as np
import nibabel as nib
import pandas as pd
import sys
import os
from numpy.testing import assert_almost_equal, assert_array_equal

'''
pathtoclassdata = "data/"

sys.path.append(os.path.join(os.path.dirname(__file__), "../code/"))

from . import scene
'''
## Need to fix the path

## Create the test image data
shape_3d = (40, 40, 20)
V = np.prod(shape_3d)
T = 438
arr_2d = np.random.normal(size=(V, T))
expected_stds = np.std(arr_2d, axis=0)
arr_4d = np.reshape(arr_2d, shape_3d + (T,))

scenes = pd.read_csv('scene_times_nums.csv', header = None) 
scenes = scenes.values

TR = 2
NUM_VOLUMES = arr_4d.shape[-1] 
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
ALL_IDS = list(range(1, 91))

#############################################################################
    
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


def test_on_off_course():
    f1 = on_off_course([26],factor_grid)
    r1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0])
    assert_almost_equal(f1,r1)

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

def test_multiple_factors_course():
    f2 = multiple_factors_course([66], factor_grid)
    r2 =np.array([66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
       66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
       66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
       66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
       66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
       66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
       66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
       66, 66, 66, 66, 66, 66, 66, 66, 66,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0, 66, 66, 66, 66, 66,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0, 66, 66, 66, 66, 66, 66, 66,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 66, 66, 66])
    assert_almost_equal(f2,r2)



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


test_GUMP_SCENES_IDS = [26, 36, 25, 38]
samp_gump, miss_gump = gen_sample_by_factors(test_GUMP_SCENES_IDS, factor_grid, False)
 
def test_gen_sample_by_factors():
    test_GUMP_SCENES_IDS = [26, 36, 25, 38]
    g1=gen_sample_by_factors(test_GUMP_SCENES_IDS, factor_grid, False)
    f3=g1[0].values()[0]
    r3=(np.array([128, 129, 130, 131, 132, 133, 134, 135, 136, 137]),
        np.array([138, 139, 140, 141, 142, 143, 144, 145, 146, 147]))
    f4=g1[1]
    r4=[25]
    assert_almost_equal(f3,r3)
    assert_almost_equal(f4,r4)

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

 
def test_training_samples():
    r5 = np.array([128, 129, 130, 131, 132, 133, 134, 135, 136, 137])
    f5= get_training_samples(samp_gump).values()[0]
    assert_almost_equal(f5,r5)
    
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
    
def test_make_label_by_time():
    g2= make_label_by_time(get_training_samples(samp_gump))
    r2 =(np.array([26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 36, 36, 36, 36, 36, 36, 36,
        36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 38, 38, 38, 38,
        38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
        38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
        38, 38, 38, 38, 38]),
 np.array([128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 148, 149, 150,
        164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
        177, 178, 179, 180, 185, 186, 187, 188, 189, 190, 191, 192, 193,
        194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
        207, 208, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
        261, 262, 263, 264, 265, 266, 267, 268]))
    assert_almost_equal(g2,r2)
    

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
    
def test_other_scene_ids():
    test_GUMP_SCENES_IDS = [26, 36, 25, 38]
    f6= other_scene_ids(test_GUMP_SCENES_IDS)
    similar = []
    for tup in test_GUMP_SCENES_IDS:
        if tup in f6:
            similar.append(tup)
    r6 =[]
    assert_almost_equal(similar,r6)
    


    