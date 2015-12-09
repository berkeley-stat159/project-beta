
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
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.cluster import KMeans
                                            
#All file strings corresponding to BOLD data for subject 4 
files = ['../data/task001_run001.bold_dico.nii', '../data/task001_run002.bold_dico.nii', 
         '../data/task001_run003.bold_dico.nii', '../data/task001_run004.bold_dico.nii', 
         '../data/task001_run005.bold_dico.nii', '../data/task001_run006.bold_dico.nii',
         '../data/task001_run007.bold_dico.nii', '../data/task001_run008.bold_dico.nii']

#Use only 6810 of the voxels (choose most variance-stable voxels for further analysis)
SUBSET_VOXELS_PATH = '../brain_mask/mask6810.npy'
mask = np.load(SUBSET_VOXELS_PATH) #mask 
VOXEL_INDCS = np.where(mask == True)[0]

unraveled = []
for one_d_pos in VOXEL_INDCS:
    three_d_pos = np.unravel_index(one_d_pos, (160, 160, 36))
    unraveled.append(three_d_pos)
unraveled = np.array(unraveled)

first_comp = unraveled[:,0]
sec_comp = unraveled[:,1]
third_comp = unraveled[:,2]

#Load in data 
all_data = []
for index, filename in enumerate(files):
    new_data = dl.load_data(filename) #load_data function drops first 4 for us
    new_data = new_data[first_comp, sec_comp, third_comp, :] #vox by time array 2d
    num_vols = new_data.shape[-1]
    if index != 0 and index != 7:
        new_num_vols = num_vols - 4   
        new_data = new_data[:,:new_num_vols] #Drop last 4 volumes for middle runs    
    all_data.append(new_data)

scenes_path = '../data/scene_times_nums.csv'
scenes = pd.read_csv(scenes_path, header = None) 
scenes = scenes.values #Now just a numpy array

combined_runs = np.concatenate(all_data, axis = 1) 
combined_runs = combined_runs[:,9:] #First 17 seconds are credits/no scene id so drop
all_data = [] #Dont need to store this anymore

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

############ K-means Analysis #################################

#Comparison between Military and Gump Scenes 
all_ids_1 = GUMP_SCENES_IDS + MILITARY_IDS 
samp_1 = sn.all_factors_indcs(all_ids_1, factor_grid)
train_labs1, train_times1 = sn.make_label_by_time(samp_1)
milt_subarr = combined_runs[:,train_times1]

kmeans = KMeans(n_clusters=2, n_init=10)
pred1 = kmeans.fit_predict(milt_subarr.T)

#Check accuracy 
#Make a vector that is 1 for Gump Scenes and 0 otherwise 
on_off_1 = sn.on_off_course(GUMP_SCENES_IDS, train_labs1)
num = sn.analyze_performance(pred1, on_off_1)
accuracy1 = max(num, 1 - num)  #88% accuracy 

#Comparison between Gump, School, Political, Military, Outside Scenes
all_ids_2 = GUMP_SCENES_IDS + SCHOOL + MILITARY_IDS 
samp_2 = sn.all_factors_indcs(all_ids_2, factor_grid)
lab2, times2 = sn.make_label_by_time(samp_2)
subarr2 = combined_runs[:,times2]

kmeans = KMeans(n_clusters=3, n_init=10)
pred2 = kmeans.fit_predict(subarr2.T)

#Set up categories: gump = 0, school = 1, military = 2,
lab_course2 = []
for val in lab2:
    if val in GUMP_SCENES_IDS:
        lab_course2.append(0)
    if val in SCHOOL:
        lab_course2.append(1)
    if val in MILITARY_IDS:
        lab_course2.append(2)

#Asses performance - harder because multiple categories 
def analyze_mult_factors(pred_labels, divided_cat_lst, num_labels):
    all_perms = list(itertools.permutations(range(num_labels)))
    relative_perform = []
    total_perform = []
    for perm in all_perms:
        props = []
        total = 0
        for index, lab in enumerate(perm):
            num_matches = np.where(divided_cat_lst[index] == lab)[0].shape[0]
            prop_hits = num_matches / divided_cat_lst[index].shape[0]
            total += num_matches
            props.append(prop_hits)
        total_perform.append(total)
        relative_perform.append(props)
    return (total_perform, relative_perform, all_perms)

def get_max(analysis_factors, num_labs):
    max_val = max(analysis_factors[0])
    max_ind = [i for i, j in enumerate(analysis_factors[0]) if j == max_val]
    max_ind = max_ind[0]
    return (analysis_factors[0][max_ind]/num_labs, analysis_factors[1][max_ind], analysis_factors[2][max_ind])

lab_course2 = np.array(lab_course2)

gump_incs = np.where(lab_course2 == 0)
school_indcs = np.where(lab_course2 == 1)
military_inds = np.where(lab_course2 == 2)

gump_pred = pred2[gump_incs] #77.3% using function below 
school_pred = pred2[school_indcs] #77.4%  
military_pred = pred2[military_inds] #75%  

#Looking at all categories 
all_ids_3 = GUMP_SCENES_IDS + SCHOOL + MILITARY_IDS + SAVANNA + POLITICAL + OUTSIDE

samp_3 = sn.all_factors_indcs(all_ids_3, factor_grid)
lab3, times3 = sn.make_label_by_time(samp_3)
subarr3 = combined_runs[:,times3]

kmeans = KMeans(n_clusters=6, n_init=10)
pred3 = kmeans.fit_predict(subarr3.T)

lab_course3 = []
for val in lab3:
    if val in GUMP_SCENES_IDS:
        lab_course3.append(0)
    if val in SCHOOL:
        lab_course3.append(1)
    if val in MILITARY_IDS:
        lab_course3.append(2)
    if val in SAVANNA:
        lab_course3.append(3) 
    if val in POLITICAL:
        lab_course3.append(4)
    if val in OUTSIDE:
        lab_course3.append(5) 

lab_course3 = np.array(lab_course3)

gump_incs3 = np.where(lab_course3 == 0)
school_indcs3 = np.where(lab_course3 == 1)
military_inds3 = np.where(lab_course3 == 2)
savanna_incs3 = np.where(lab_course3 == 3)
political_indcs3 = np.where(lab_course3 == 4)
outisde_inds3 = np.where(lab_course3 == 5)

gump_pred3 = pred3[gump_incs3]  
school_pred3 = pred3[school_indcs3] 
military_pred3 = pred3[military_inds3]
savanna_pred3 = pred3[savanna_incs3]
political_pred3 = pred3[political_indcs3]
outisde_pred3 = pred3[outisde_inds3]
combined3 = [gump_pred3, school_pred3, military_pred3, savanna_pred3, political_pred3, outisde_pred3]

analysis_fact3 = analyze_mult_factors(pred3, combined3, 6) #51% overall  
performance3 = get_max(analysis_fact3, pred3.shape[0]) #Military and political scenes seem correlated
                                                       #Gump and outside scenes seem correlated 

#Since Military and Political Scenes seem correlated lets see how we do combining them 
all_ids_4 = GUMP_SCENES_IDS + SCHOOL + MILITARY_IDS +  POLITICAL + SAVANNA  
samp_4 = sn.all_factors_indcs(all_ids_4, factor_grid)
lab4, times4 = sn.make_label_by_time(samp_4)
subarr4 = combined_runs[:,times4]

kmeans = KMeans(n_clusters=4, n_init=10)
pred4 = kmeans.fit_predict(subarr4.T)

lab_course4 = []
for val in lab4:
    if val in GUMP_SCENES_IDS:
        lab_course4.append(0)
    if val in SCHOOL:
        lab_course4.append(1)
    if val in MILITARY_IDS or val in POLITICAL:
        lab_course4.append(2)
    if val in SAVANNA:
        lab_course4.append(3) 

lab_course4 = np.array(lab_course4)

gump_incs4 = np.where(lab_course4 == 0)
school_indcs4 = np.where(lab_course4 == 1)
military_pol_inds4 = np.where(lab_course4 == 2)
savanna_incs4 = np.where(lab_course4 == 3)

gump_pred4 = pred4[gump_incs4]  
school_pred4 = pred4[school_indcs4] 
military_pol_pred4 = pred4[military_pol_inds4]
savanna_pred4 = pred4[savanna_incs4]
combined4 = [gump_pred4, school_pred4, military_pol_pred4, savanna_pred4]

analysis_fact4 = analyze_mult_factors(pred4, combined4, 4)
performance4 = get_max(analysis_fact4, pred4.shape[0]) #62% overall  

#smooth spatially - voxels next to each other are not iid - some filter (gaussian best) to fix this 


