import numpy as np
import os
import sys
from numpy.testing import assert_almost_equal, assert_array_equal


__file__ = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../utils/")))

from vmt_utils import *

#create a matrix and a vector
mtx = np.arange(9).reshape((3,3))
d = np.arange(3)

#create a dumpy feature space
stim_fs_est = np.zeros((20, 1155))
tmp = add_lags(stim_fs_est, [2,3,4])



def test_mult_diag():
	exp = np.dot(np.diag(d), mtx)
	actual = mult_diag(d, mtx)
	assert_almost_equal(actual, exp)

def test_add_lags():
	actual = add_lags(stim_fs_est, [2,3,4]).shape
	d2 = stim_fs_est.shape[1]*3
	exp = (20, d2)
	assert_almost_equal(actual, exp)

def test_add_contant():
	actual = add_constant(tmp,is_first=True).shape
	d3 = tmp.shape[1]+1
	exp = (20, d3)
	assert_almost_equal(actual, exp)

