import numpy as np
from ..vmt_utils import *
from numpy.testing import assert_almost_equal, assert_array_equal


# __file__ = os.getcwd()
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../utils/")))


d = np.arange(9).reshape((3,3))
mtx = np.arange(3)

def test_mult_diag():
	exp = np.dot(np.diag(d), mtx)
	actual = mult_diag(d, mtx)
	assert_almost_equal(actual, exp)

# def add_lags(stim_fs_est,[2,3,4]):