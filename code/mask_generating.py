
# coding: utf-8

###
#The code loads a subset of smoothed data and performs filtering.
#It then analyzes filtered data variance and generate mask for the full data
###

# In[5]:

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec
import nitime

# Import the time-series objects:
from nitime.timeseries import TimeSeries

# Import the analysis objects:
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer


# In[24]:

data = np.load('../data/smoothed_data.npy')
# data = np.load('../../val_runs.npy')


# In[26]:

data2d = data.reshape((data.shape[0]*data.shape[1]*data.shape[2],data.shape[3]))


# In[28]:
# plt.figure(1)
# plt.plot(data2d[87440,:])


# In[29]:

TR = 2


# In[30]:

T = TimeSeries(data2d, sampling_interval=TR)


# In[31]:

S_original = SpectralAnalyzer(T)


# In[32]:

fig01 = plt.figure(2)
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
plt.ylim((0,25))

ax01.legend()
plt.savefig("../figure/FFT.jpg")


# In[33]:

F = FilterAnalyzer(T, ub=0.15, lb=0.02)

# Initialize a figure to display the results:
fig02 = plt.figure(3)
ax02 = fig02.add_subplot(1, 1, 1)

# Plot the original, unfiltered data:
ax02.plot(F.data[87440], label='unfiltered')

# ax02.plot(F.filtered_boxcar.data[89], label='Boxcar filter')

# ax02.plot(F.fir.data[89], label='FIR')

# ax02.plot(F.iir.data[89], label='IIR')

ax02.plot(F.filtered_fourier.data[87440], label='Fourier')
ax02.legend()
ax02.set_xlabel('Time (TR)')
ax02.set_ylabel('Signal amplitude (a.u.)')

plt.savefig("../figure/data_filtering_on_smoothed_data.jpg")


# In[34]:

F.filtered_fourier.data.shape


# In[35]:

fdata = F.filtered_fourier.data


# In[36]:

fdata.shape


# In[37]:

v = np.var(fdata,axis=1)


# In[39]:
plt.figure(4)
plt.hist(v)
plt.xlabel("voxels")
plt.ylabel("variance")
plt.title("variance of the voxel activity filtering")
plt.savefig("../figure/voxel_variance_on_smoothed_data.jpg")


# In[40]:

len(v)


# In[52]:

mask55468 = v>13
print sum(mask55468)
np.save('../brain_mask/sm_mask55468.npy',mask55468)


# In[54]:

mask17887 = v>23
print sum(mask17887)
np.save('../brain_mask/sm_mask17887.npy',mask17887)


# In[56]:

mask9051 = v>33
print sum(mask9051)
np.save('../brain_mask/sm_mask9051.npy',mask9051)



