
#ADD PATHS FIX 
import os
import sys

from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import task_frame as tf

#Load data
#All file strings corresponding to BOLD data for subject 4 

files = ['task001_run001.bold_dico.nii', 'task001_run002.bold_dico.nii', 
         'task001_run003.bold_dico.nii', 'task001_run004.bold_dico.nii', 
         'task001_run005.bold_dico.nii', 'task001_run006.bold.nii'
         'task001_run007.bold.nii', 'task001_run008.bold.nii']


# Load the images as an image object
# Load all the image data from the images
# Drop the first four volumes, as we know these are outliers

all_data = []
for index, filename in enumerate(files):
	new_data = dl.load_data(filename) #load_data function drops first 4 for us
	num_vols = new_data.shape[-1]
	if index != 0 or index != 7:
		new_num_vols = num_vols - 4
		new_data = new_data[:,:,:,:new_num_vols] #Drop last 4 volumes for middle runs
	all_data.append(new_data)

#Load task frame from data folder 
task_frame =  genfromtxt('scene_times_nums.csv', delimiter=',')

#Let's look at the correlation in voxel time courses between every 'school' 
#scene as the stimulus for run 0
TR = 2
n_trs = 447 

school = tf.make_valid_task(task_frame, [67, 68, 69], 0)
neural_school = tf.events2neural(school, TR, n_trs)

house = tf.make_valid_task(task_frame, [38, 40], 0)
neural_house = tf.events2neural(school, TR, n_trs)

savanah = tf.make_valid_task(task_frame, [66], 0)
neural_savanah = tf.events2neural(savanah, TR, n_trs)

tree = tf.make_valid_task(task_frame, [73], 0)
neural_tree = tf.events2neural(savanah, TR, n_trs)

plt.figure(1)
plt.ylim(0, 1.2)
plt.plot(neural_tree, 'y', label = 'school')
plt.plot(neural_school, 'g', label = 'tree')
plt.plot(neural_house, 'b', label = 'house')
plt.plot(neural_savanah, 'r', label = 'savanah')
plt.savefig('neural_school')
plt.title('Neural Stimulus')
plt.xlabel('time')
plt.ylabel('on/off')
plt.legend(loc = 'lower right')
plt.close()

#voxel_time_course = data[42, 32, 19]
#np.corrcoef(neural, voxel_time_course)