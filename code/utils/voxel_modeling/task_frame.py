#Created By: Raj Agrawal 
#11/12/15

import numpy as np

#In order to load task_frame from data folder do the following: 
#from numpy import genfromtxt
#task_frame =  genfromtxt('scene_times_nums.csv', delimiter=',')

#Create segement times dictionary 
segements = list(range(8))
times = [(0, 894), (894, 1760), (1760, 2620), (2620, 3580), 
         (3580, 4488), (4488, 5350), (5350, 6418), (6418, 7086)]
zipped = zip(segements, times)

seg_map = {}
for key, val in zipped:
    seg_map[key] = val
   
def get_all_runs():
    return seg_map

def get_run_time(run_id):
    return get_all_runs()[run_id]

def get_run_task_frame(task_frame, run_id):
    time_interval = get_run_time(run_id)
    lower = time_interval[0]
    upper = time_interval[1]
    onset_times = task_frame[:,0]
    keep_indces = np.where(np.logical_and(onset_times>=lower, onset_times<=upper))[0]
    return task_frame[keep_indces,]

def filt_task_frame(task_frame, ids, run_id):
    """
    input: 'task_frame' - a four column numpy array with columns containing (in order) 
            onset time, duration, amplitue, factor ids (see 'scene_times_nums.csv' as 
            an example format)

           'run_id' - the movie segement portion to include (i.e. which rows of 
            'task frame') to include 
           
           'ids' - a list of ids which should be included in the filtered task frame
    
    returns: a four-column numpy array that only has the specified factor ids and is 
             only contains rows within the interval associated with the 'run_id'

    """
    assert run_id in segements
    assert type(ids) == list
    filt_task_frame = get_run_task_frame(task_frame, run_id)
    factor_ids = filt_task_frame[:, 3]
    indcs = []
    for i in ids:
        tup = np.where(factor_ids == i)
        arr = tup[0]
        indcs.extend(arr)
        indcs.sort()
    return filt_task_frame[indcs, ]

def make_valid_task(task_frame, ids, run_id):
    """
    input: 'task_frame' - a four column numpy array with columns containing (in order) 
            onset time, duration, amplitue, factor ids

            'run_id' - the movie segement portion to include (i.e. which rows of 
            'task frame') to include
    
    returns: a three-column numpy array (onset time, duration, amplitue) that only has 
             the specified factor ids to be used in event2neural 
    """
    filt_frame = filt_task_frame(task_frame, ids, run_id)
    return filt_frame[:,:3]

def events2neural(task_frame, tr, n_trs):
    """ Return predicted neural time course from event file `task_fname`

    Parameters
    ----------
    task_frame : numpy array 
       task_frame array 
    tr : float
        TR in seconds
    n_trs : int
        Number of TRs in functional run

    Returns
    -------
    time_course : array shape (n_trs,)
        Predicted neural time course, one value per TR
    """
    # Check that the file is plausibly a task file
    if task_frame.ndim != 2 or task_frame.shape[1] != 3:
        raise ValueError("Is {0} really a task file?", tr)
    # Convert onset, duration seconds to TRs
    task_frame[:, :2] = task_frame[:, :2] / tr
    # Neural time course from onset, duration, amplitude for each event
    time_course = np.zeros(n_trs)
    for onset, duration, amplitude in task_frame:
        time_course[onset:onset + duration] = amplitude
    return time_course 
