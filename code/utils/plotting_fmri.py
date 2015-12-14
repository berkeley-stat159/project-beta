
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt

""" In this module, we have provided plotting functions to help visulize fMRI data."""

def plot_grayscale(data):
    plt.imshow(data, cmap='gray')
    
def plot_sdevs(sdevs, outliers_sdevs, outlier_interval):
    plt.plot(sdevs, label = 'volume SD values')
    put_o = sdevs[outliers_sdevs]
    plt.plot(outliers_sdevs, put_o, 'o', label = 'outlier points')
    plt.axhline(y=outlier_interval[0], linestyle='dashed')
    plt.axhline(y=outlier_interval[1], linestyle='dashed')
    plt.xlabel('Index')
    plt.ylabel('SD')
    plt.title('Volume Standard Deviations')
    plt.legend(loc = 'lower right')

def plot_rms(rms, outliers_rms, outlier_interval):
    plt.plot(rms, label = 'RMS values')
    put_rms_o = rms[outliers_rms]
    plt.plot(outliers_rms, put_rms_o, 'o', label = 'outlier points')
    plt.axhline(y=outlier_interval[0], linestyle='dashed')
    plt.axhline(y=outlier_interval[1], linestyle='dashed')
    plt.xlabel('Index')
    plt.ylabel('RMS')
    plt.title('RMS Differences')
    plt.legend(loc='upper right')
