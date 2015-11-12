
import numpy as np

#In order to load task_frame from data folder do the following: 
#from numpy import genfromtxt
#task_frame =  genfromtxt('scene_times_nums.csv', delimiter=',') 

def filt_task_frame(task_frame, ids):
    """
    input: 'task_frame' - a four column numpy array with columns containing (in order) 
            onset time, duration, amplitue, factor ids
           
           'ids' - a list of ids which should be included in the filtered task frame
    
    returns: a four-column numpy array that only has the specified factor ids

    """
    assert type(ids) == list
    indcs = []
    factor_ids = task_frame[:, 3]
    for i in ids:
        tup = np.where(factor_ids == i)
        arr = tup[0]
        indcs.extend(arr)
    return task_frame[indcs, ]

def make_valid_task(task_frame, ids):
    """
    input: 'task_frame' - a four column numpy array with columns containing (in order) 
            onset time, duration, amplitue, factor ids
    
    returns: a three-column numpy array (onset time, duration, amplitue) that only has 
             the specified factor ids to be used in event2neural 
    """
    filt_frame = filt_task_frame(task_frame, ids)
    return filt_frame[:,:3] 
