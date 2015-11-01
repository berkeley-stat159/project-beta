
from __future__ import division, print_function, absolute_import
import data_loading 

""" In this module, we have provided plotting functions to help visulize fMRI data."""

def plot_grayscale(data):
	""" 
	Returns a grayscale 

	""" 
    plt.imshow(data, cmap='gray')


def plot_removed_noise(data, axis, lower=.05, upper=.95):
	new_data = remove_outliers(data, axis, lower, upper)
	plot_grayscale(new_data)


def plot_mask(data):



#time-series plots 