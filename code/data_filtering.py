import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import nitime

# Import the time-series objects:
from nitime.timeseries import TimeSeries

# Import the analysis objects:
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer

os.getcwd()
os.chdir('..')
os.chdir('data')

## load data
data2d = np.load('masked_data_50k.npy')


## warning
print('Warning!! This scripts take at least 20 minutes to run.')

## plot data
plt.plot(data2d[7440,:])

## setting the TR 
TR = 2

T = TimeSeries(data2d, sampling_interval=TR)

## examining the spectrum of the original data, before filtering. 
# We do this by initializing a SpectralAnalyzer for the original data:
S_original = SpectralAnalyzer(T)

fig01 = plt.figure()
ax01 = fig01.add_subplot(1, 1, 1)
# ax01.plot(S_original.psd[0],
#           S_original.psd[1][9],
#           label='Welch PSD')

ax01.plot(S_original.spectrum_fourier[0],
          np.abs(S_original.spectrum_fourier[1][9]),
          label='FFT')

ax01.plot(S_original.periodogram[0],
          S_original.periodogram[1][9],
          label='Periodogram')

# ax01.plot(S_original.spectrum_multi_taper[0],
#           S_original.spectrum_multi_taper[1][9],
#           label='Multi-taper')

ax01.set_xlabel('Frequency (Hz)')
ax01.set_ylabel('Power')
plt.ylim((0,8000))

ax01.legend()
plt.savefig("../figure/FFT.jpg")
print('FFT.jpg saved')

## We start by initializing a FilterAnalyzer. 
#This is initialized with the time-series containing the data 
#and with the upper and lower bounds of the range into which we wish to filter

F = FilterAnalyzer(T, ub=0.15, lb=0.02)

# Initialize a figure to display the results:
fig02 = plt.figure()
ax02 = fig02.add_subplot(1, 1, 1)

# Plot the original, unfiltered data:
ax02.plot(F.data[7440], label='unfiltered')
ax02.plot(F.filtered_fourier.data[7440], label='Fourier')
ax02.legend()
ax02.set_xlabel('Time (TR)')
ax02.set_ylabel('Signal amplitude (a.u.)')

plt.savefig("../figure/data_filtering_on_smoothed_data.jpg")
print('data_filtering_on_smoothed_data.jpg')

np.save('filtered_data.npy',F.filtered_fourier.data)
print('filtered_data.npy saved')

F.filtered_fourier.data.shape

fdata = F.filtered_fourier.data

fdata.shape

v = np.var(fdata,axis=1)

plt.hist(v)
plt.xlabel("voxels")
plt.ylabel("variance")
plt.title("variance of the voxel activity filtering")
plt.savefig("../figure/voxel_variance_on_smoothed_data.jpg")
print('voxel_variance_on_smoothed_data.jpg')