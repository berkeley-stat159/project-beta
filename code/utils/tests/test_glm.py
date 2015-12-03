import os
import sys
import numpy as np
from sklearn import linear_model as lm 
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
from numpy.testing import assert_almost_equal, assert_array_equal

__file__ = os.getcwd()

convolved = np.loadtxt("ds114_sub009_t2r1_conv.txt")[4:]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../utils/")))
from glm import *

## Create image data
shape_3d = (64, 64, 30)
V = np.prod(shape_3d)
T = 169
arr_2d = np.random.normal(size=(V, T))
expected_stds = np.std(arr_2d, axis=0)
data = np.reshape(arr_2d, shape_3d + (T,))


def test_glm():
    actual_design = np.ones((len(convolved), 2))
    actual_design[:, 1] = convolved
    exp_design , exp_B_4d = glm(data, convolved)
    assert_almost_equal(actual_design, exp_design)

def test_glm1():
    actual_design = np.ones((len(convolved), 2))
    actual_design[:, 1] = convolved
    data_2d = np.reshape(data, (-1, data.shape[-1]))
    actual_B = npl.pinv(actual_design).dot(data_2d.T)
    actual_B_4d = np.reshape(actual_B.T, data.shape[:-1] + (-1,))
    exp_design , exp_B_4d,  = glm(data, convolved)
    assert_almost_equal(actual_B_4d, exp_B_4d)


def test_scale_design_mtx():
    actual_design = np.ones((len(convolved), 2))
    actual_design[:, 1] = convolved
    exp_design , exp_B_4d = glm(data, convolved)
    f1=scale_design_mtx(exp_design)[-1]
    r1=np.array([ 1.        ,  0.16938989])
    assert_almost_equal(f1,r1)