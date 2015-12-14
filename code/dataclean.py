# 10/29/15 
# Cindy Mo  
# clean data from translated text file 
# remove stop words 

from collections import OrderedDict
import csv 
import numpy as np 
import pandas as pd 
import re
import json
import nltk


# data file
description = open("../data/description.csv") 
csv_description = csv.reader(description) 

# stopwords file
stopwords = open("../data/stopwords.txt")
stopwords_list = stopwords.read().splitlines()

# extract set of all significant words from text
with open("../data/description.csv") as f:
	uwords = set()
	for row in f:
		r = row[3].lower().split()
		r = [re.sub(r'\W+', '', w) for w in r if re.sub(r'\W+', '', w) != ""]
		r = [w for w in r if w not in stopwords_list]
		uwords |= set(r)
	with open("../data/uniquewords.json", 'w') as f:
		json.dump(dict.fromkeys(uwords), f) 

affordance_dict_fpath = "../data/uniquewords.json" #loadin the json file with all unique words as keys

with open(affordance_dict_fpath) as fid:
    affordanceDict = json.loads(fid.readlines()[0])
fid.close()

all_words = affordanceDict.keys()
word_dict = dict.fromkeys(all_words)



def get_wn_synsets(lemma):
   """
   Get all synsets for a word, return a list of [wordnet_label,definition, hypernym_string]
   for all synsets returned.
   """
   from nltk.corpus import wordnet as wn
   synsets = wn.synsets(lemma)
   out = []
   for s in synsets:
       # if not '.v.' in s.name(): continue # only verbs!
       hyp = ''
       for ii,ss in enumerate(s.hypernym_paths()):
           try:
               hyp+=(repr([hn.name() for hn in ss])+'\n')
           except:
               hyp+='FAILED for %dth hypernym\n'%ii
       out.append(dict(synset=s.name(), definition=s.definition(),hypernyms=hyp))
   return out

def get_wn_meaning(lemma):
    """get meaning of a word using wordNet labels"""
    # from nltk.corpus import wordnet as wn
    # return wn.synset(lemma).definition()
    return None


for w in all_words:
    if get_wn_synsets(w) == []:
        word_dict.pop(w)
    else:
        word_dict[w] = [get_wn_synsets(w)[0]['synset']]

with open("../data/word2wn.json", 'w') as f:
    json.dump(word_dict, f) 


with open("../data/word2wn.json") as f:
	translator = json.loads(f.read())
for w in translator:
	translator[w] = translator[w][0]
print translator

with open("../data/description.csv") as f:
	storedictduration = OrderedDict()
	storedictstop = OrderedDict()
	ind = 0
	for row in csv_description: 
		if ind == 0:
			pass
		else:
			start, stop, duration = row[0], row[1], str(float(row[1])-float(row[0]))
			raw = row[3].lower().split()
			refined = [re.sub(r'\W+', '', w) for w in raw if re.sub(r'\W+', '', w) != ""]
			refined = [translator[w] for w in refined if w in translator]
			storedictduration[ind] = {"start":start, "duration":duration, "words":refined}
			storedictstop[ind] = {"start":start, "stop":stop, "words":refined}
		ind += 1
	with open("../data/wordnet_stop.json", 'w') as f:
		json.dump(storedictstop.values(), f)
	with open("../data/wordnet_duration.json", 'w') as f:
		json.dump(storedictduration.values(), f)

# print sum([len(e["words"]) for e in storedict])/float(len(storedict))

# df = pd.DataFrame(storedict.values()[1:], columns = ["start", "stop", "words"])
# df.to_csv("cleaned.csv")
