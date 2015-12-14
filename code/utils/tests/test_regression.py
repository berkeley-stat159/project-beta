# import numpy as np
# import os
# import sys
# from numpy.testing import assert_almost_equal, assert_array_equal


# __file__ = os.getcwd()
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../utils/")))
# from regression import *


# n_splits = 10 # number of subdivisions of validation data for cross validation of ridge parameter (alpha)
# n_resamps = 10 # number of times to compute regression & prediction within training data (can be <= n_splits)
# chunk_sz = 6000 # number of voxels to fit at once. Memory-saving.
# pthr = 0.005 # Ridge parameter is chosen based on how many voxels are predicted above a correlation threshold 
#              # for each alpha value (technically it's slightly more complicated than that, see the code). 
#              # This p value sets that correlation threshold.
# def test_ridge_cv():
# 	out = regression.ridge_cv(efs,data_est_masked,val_fs=vfs,
# 		val_data=data_val_masked,alphas=alpha,n_resamps=n_resamps,
#         n_splits=n_splits,chunk_sz=chunk_sz,pthr=pthr,is_verbose=True)
# 	