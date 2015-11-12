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

# data file
description = open("description.csv") 
csv_description = csv.reader(description) 

# stopwords file
stopwords = open("stopwords.txt")
stopwords_list = stopwords.read().splitlines()

# extract set of all significant words from text
uwords = set()
for row in csv_description:
	r = row[3].lower().split()
	r = [re.sub(r'\W+', '', w) for w in r if re.sub(r'\W+', '', w) != ""]
	r = [w for w in r if w not in stopwords_list]
	uwords |= set(r)
with open("uniquewords.json", 'w') as f:
	json.dump(dict.fromkeys(uwords), f) 


storedict = OrderedDict()
ind = 0
for row in csv_description: 
	start, stop = row[0], row[1]
	raw = row[3].lower().split()
	refined = [re.sub(r'\W+', '', w) for w in raw if re.sub(r'\W+', '', w) != ""]
	refined = [w for w in refined if w not in stopwords_list]
	storedict[ind] = {"start":start, "stop":stop, "words":refined}
	ind += 1

with open("cleandescription.json", 'w') as f:
	json.dump(storedict.values(), f)

# df = pd.DataFrame(storedict.values()[1:], columns = ["start", "stop", "words"])
# df.to_csv("cleaned.csv")
