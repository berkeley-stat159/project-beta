import os
import sys
import numpy as np
from sklearn import linear_model as lm 
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
from glm import *
from numpy.testing import assert_almost_equal

pathtoclassdata = "../testdata/"

sys.path.append("../prediction")

def test_glm():
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    convolved = np.loadtxt(pathtoclassdata + "ds114_sub009_t2r1_conv.txt")[4:]
    actual_design = np.ones((len(convolved), 2))
    actual_design[:, 1] = convolved
    
    data_2d = np.reshape(data, (-1, data.shape[-1]))
    actual_B = npl.pinv(actual_design).dot(data_2d.T)
    actual_B_4d = np.reshape(actual_B.T, img.shape[:-1] + (-1,))
    
    exp_design , exp_B_4d = glm(data, convolved)
    assert_almost_equal(actual_B_4d, exp_B_4d)
    assert_almost_equal(actual_design, exp_design)
