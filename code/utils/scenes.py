
#Import libraries
from __future__ import print_function, division
import numpy as np

import data_loading as dl
import numpy.linalg as npl

from scipy.spatial.distance import hamming 


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
    for factor, factor_indcs in all_factors.items():
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
    for factor, sample in samples.items():
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

def analyze_performance(predicted_labels, actual_labels):
    """ Returns the proportion of total labels that are acurately matched 
    
    Parameters
    ----------
    predicted_labels : 1d array
        Consists of numeric labels   
    actual_labels: 1d array 
        Consists of numeric labels     
    Returns
    -------
    distance : float
        'Distance' here equals #matching entries / length(predicted_labels)
    """
    normed_distance = hamming(predicted_labels, actual_labels) #between 0 - 1
    return 1 - normed_distance
