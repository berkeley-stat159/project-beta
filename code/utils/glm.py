import numpy as np
from sklearn import linear_model as lm 
import numpy.linalg as npl
import matplotlib.pyplot as plt

def glm(data_4d, convolved):
    """
    Return a tuple of the estimated coefficients in 4 dimensions and 
    the design matrix. 
    
    Parameters
    ----------
    data_4d: numpy array of 4 dimensions 
        The image data of one subject
    conv: numpy array of 1 dimension
        The convolved time course
    Note that the fourth dimension of `data_4d` (time or the number 
    of volumes) must be the same as the length of `convolved`. 
    
    Returns
    -------
    glm_results : tuple
        Estimated coefficients in 4 dimensions and the design matrix.
    """
    num_vols = data_4d.shape[-1]
    assert(len(convolved) == num_vols)
    design = np.ones((num_vols, 2))
    design[:, 1] = convolved
    vox_by_time = np.reshape(data_4d, (-1, num_vols))
    betas = npl.pinv(design).dot(vox_by_time.T)
    betas_4d = np.reshape(betas.T, data_4d.shape[:-1] + (-1,))
    return (design, betas_4d)

def scale_design_mtx(X):
    """ 
    Return a scaled design matrix for display
    
    Parameters
    ----------
    X: Design Matrix
    
    Returns
    -------
    Xs: a scaled design matrix 

    Estimated coefficients in 4 dimensions and the design matrix.
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
               cmap='hot',
               aspect='auto') # Gray colormap
    plt.xlabel("features (words)")
    plt.ylabel("time (TR)")
    plt.title(design_title)
    # plt.show()
