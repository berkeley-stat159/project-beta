###
# The code loads words in the audio description and generate design matrix
# (Time by feature). Entry in position ij indicate whether the word j appears in 
# time point i.
###



# In[108]:
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from decimal import *
import pickle



# In[72]:

fpath = '../description_pp/wordnet_stop.json'
with open(fpath) as fid:
    sti_dict_list = json.loads(fid.readlines()[0])
fid.close()


# In[87]:

all_words = []
for d in sti_dict_list:
    all_words += d['words']


# In[138]:

all_unique_words = list(set(all_words))
len(all_unique_words)


# In[139]:

#design matrix with all words at all time
# 3543+1 is the total number of TRs
design_1 = np.zeros((3544,len(set(all_words))))
for d in sti_dict_list:
    start = np.floor(float(d['start']))//2
    end = np.ceil(float(d['stop']))//2 + 1
    for tr in range(int(start),int(end)+1):
        for w in d['words']:
            idx = all_unique_words.index(w)
            design_1[tr][idx] += 1


# In[144]:
#save the design matrix
design_1_fpath = "../description_pp/design_matrix_1.npy"
np.save(design_1_fpath,design_1)

#save the figure
plt.figure()
plt.imshow(design_1,aspect='auto',cmap='hot')
plt.colorbar()
plt.savefig('../figure/visual_design_matrix.png')


#save the word list to file
word_list_fpath = "../description_pp/word_list.p"
pickle.dump(all_unique_words, open(word_list_fpath, "wb" ))