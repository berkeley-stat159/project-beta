

#Import standard libraries
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import data_loading as dl
import plotting_fmri as plt_fmri
import save_files as sv

#All file strings corresponding to BOLD data for subject 4 

files = ['task001_run001.bold_dico.nii', 'task001_run002.bold_dico.nii', 
         'task001_run003.bold_dico.nii', 'task001_run004.bold_dico.nii', 
         'task001_run005.bold_dico.nii', 'task001_run006.bold.nii'
         'task001_run007.bold.nii', 'task001_run008.bold.nii']

scenes = 'scene_times_nums.csv' 

#We will first check if there are any noticable differences between scenes
#occuring in 'Gump house/room/etc.' vs. all other scenes 

#In this simple example, we let 1 denote a Gump scene and 0 for all other 
#scenes 

#FIX: PUT THESE FUNCTIONS IN UTILS 

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

GUMP_SCENES_IDS = [38, 40, 41, 42] #factor ids of Gump scenes
OTHER_SCENES = 


#Next we will look at political scenes vs. non-political scenes 

#Here we investigate outdoors vs. indoor scenes 

#The following compares the 6 major scenes in the film 