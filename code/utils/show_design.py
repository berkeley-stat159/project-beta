from glm import *
import matplotlib.pyplot as plt

design_fpath = "design_matrix_1.npy"
X = np.load(design_fpath)
show_design(X, 'design matrix')
plt.savefig('visual_design_matrix.png')
