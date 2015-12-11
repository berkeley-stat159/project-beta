# created by Cindy 
# 12/8/15 

""" Tests for data_loading module 
Run with: 
    nosetests test_data_loading.py
"""
from __future__ import print_function 
import os 
import sys 
import numpy as np 
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal 

__file__ = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../utils/")))


from data_loading import * 

# create a 4D test Numpy array 
test_array = np.random.randn(840).reshape((4, 5, 6, 7)) 

def test_vox_by_time(): 
    n_voxels = 4*5*6 
    test_reshape = test_array.reshape(n_voxels, 7) 
    actual_reshape = vox_by_time(test_array)
    assert_array_equal(test_reshape, actual_reshape)    

def test_vol_std(): 
	test_vol_std = np.std(np.reshape(test_array, (120, 7)), axis = 0)
	actual_vol_std = vol_std(test_array)
	assert_array_equal(test_vol_std, actual_vol_std)

def test_iqr_outliers():
	 # Test with simplest possible array
    arr = np.arange(101)  # percentile same as value
    # iqr = 50
    exp_lo = 25 - 75
    exp_hi = 75 + 75
    indices, thresholds = iqr_outliers(arr)
    assert_array_equal(indices, [])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Reverse, same values
    indices, thresholds = iqr_outliers(arr[::-1])
    assert_array_equal(indices, [])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Add outliers
    arr[0] = -51
    arr[1] = 151
    arr[100] = 1  # replace lost value to keep centiles same
    indices, thresholds = iqr_outliers(arr)
    assert_array_equal(indices, [0, 1])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Reversed, then the indices are reversed
    indices, thresholds = iqr_outliers(arr[::-1])
    assert_array_equal(indices, [99, 100])
    assert_equal(thresholds, (exp_lo, exp_hi))


