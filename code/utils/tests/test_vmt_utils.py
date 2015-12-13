import numpy as np
import os
import sys
from numpy.testing import assert_almost_equal, assert_array_equal


__file__ = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../utils/")))
from vmt_utils import *

mtx = np.arange(9).reshape((3,3))
d = np.arange(3)

# fs = 

def test_mult_diag():
	exp = np.dot(np.diag(d), mtx)
	actual = mult_diag(d, mtx)
	assert_almost_equal(actual, exp)

# def add_lags(stim_fs_est,[2,3,4]):
