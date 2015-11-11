
import numpy as np
from sklearn import linear_model as lm 
import numpy.linalg as npl

def sk_regression(data_4d, convolved, model):
	num_vols = data_4d.shape[-1]
	assert(len(convolved) == num_vols)
	X = np.ones(num_vols, 2))
	X[:, 1] = convolved
	vox_by_time = np.reshape(data_4d, (-1, num_vols))

def glm(data_4d, convolved):
	num_vols = data_4d.shape[-1]
	assert(len(convolved) == num_vols)
	design = np.ones((num_vols, 2))
	design[:, 1] = convolved
	vox_by_time = np.reshape(data_4d, (-1, num_vols))
	betas = npl.pinv(design).dot(vox_by_time.T)
	betas_4d = np.reshape(betas.T, data_4d.shape[:-1] + (-1,))
	return (X, betas_4d)

def scale_design_mtx(X):
    """utility to scale the design matrix for display

    This scales the columns to their own range so we can see the variations
    across the column for all the columns, regardless of the scaling of the
    column.
    """
    mi, ma = X.min(axis=0), X.max(axis=0)
    # Vector that is True for columns where values are not
    # all almost equal to each other
    col_neq = (ma - mi) > 1.e-8
    Xs = np.ones_like(X)
    # Leave columns with same value throughout with 1s
    # Scale other columns to min, max in column
    mi = mi[col_neq]
    ma = ma[col_neq]
    Xs[:,col_neq] = (X[:,col_neq] - mi)/(ma - mi)
    return Xs

def show_design(X, design_title):
    """ Show the design matrix nicely """
    plt.imshow(scale_design_mtx(X),
               interpolation='nearest',
               cmap='gray') # Gray colormap
    plt.title(design_title)
