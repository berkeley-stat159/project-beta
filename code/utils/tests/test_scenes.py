from __future__ import print_function, division
import numpy as np
import nibabel as nib
import pandas as pd
import sys
import os
import six
from numpy.testing import assert_almost_equal, assert_array_equal

#uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
__file__ = os.getcwd()

#sys.path.append(uppath(__file__, 1))
sys.path.append(os.path.join(os.path.dirname(__file__),"../utils/"))
from scenes import on_off_course, multiple_factors_course,gen_sample_by_factors,get_training_samples,get_tst_samples,make_label_by_time,other_scene_ids,analyze_performance

## Create the test image data
shape_3d = (40, 40, 20)
V = np.prod(shape_3d)
T = 438
arr_2d = np.random.normal(size=(V, T))
expected_stds = np.std(arr_2d, axis=0)
arr_4d = np.reshape(arr_2d, shape_3d + (T,))


datapath = '../../../data/'
scenes = pd.read_csv(datapath+'scene_times_nums.csv', header = None) 
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


test_GUMP_SCENES_IDS = [26, 36, 25, 38]
samp_gump, miss_gump = gen_sample_by_factors(test_GUMP_SCENES_IDS, factor_grid, False)
 
def test_gen_sample_by_factors():
    test_GUMP_SCENES_IDS = [26, 36, 25, 38]
    g1=gen_sample_by_factors(test_GUMP_SCENES_IDS, factor_grid, False)
    f3=list(g1[0].values())[0]
    r3=(np.array([128, 129, 130, 131, 132, 133, 134, 135, 136, 137]),
        np.array([138, 139, 140, 141, 142, 143, 144, 145, 146, 147]))
    f4=g1[1]
    r4=[25]
    assert_almost_equal(f3,r3)
    assert_almost_equal(f4,r4)

def test_get_training_samples():
    r5 = np.array([128, 129, 130, 131, 132, 133, 134, 135, 136, 137])
    f5= list(get_training_samples(samp_gump).values())[0]
    assert_almost_equal(f5,r5)
    
def test_get_tst_samples():
    f7 =list(get_tst_samples(samp_gump).values())[1]
    r7 =np.array([181, 182, 183, 184, 327, 328, 329, 330, 331, 332, 333, 334, 335,
       336, 337, 338, 339, 340, 341, 342])
    assert_almost_equal(f7,r7)
    
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
    
def test_other_scene_ids():
    test_GUMP_SCENES_IDS = [26, 36, 25, 38]
    f6= other_scene_ids(test_GUMP_SCENES_IDS)
    similar = []
    for tup in test_GUMP_SCENES_IDS:
        if tup in f6:
            similar.append(tup)
    r6 =[]
    assert_almost_equal(similar,r6)
    

def test_analyze_performance():
    predicted_labels=np.array([26,27,28,78,66,39])
    actual_labels=np.array([26,38,39,78,39,29])
    f8 =analyze_performance(predicted_labels, actual_labels)
    r8 = 0.33333333333333337
    assert_almost_equal(f8,r8)