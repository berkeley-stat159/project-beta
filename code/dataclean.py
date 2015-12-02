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
with open("description.csv") as f:
	uwords = set()
	for row in f:
		r = row[3].lower().split()
		r = [re.sub(r'\W+', '', w) for w in r if re.sub(r'\W+', '', w) != ""]
		r = [w for w in r if w not in stopwords_list]
		uwords |= set(r)
	with open("uniquewords.json", 'w') as f:
		json.dump(dict.fromkeys(uwords), f) 

with open("word2wn.json") as f:
	translator = json.loads(f.read())
for w in translator:
	translator[w] = translator[w][0]
print translator

with open("description.csv") as f:
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
	with open("wordnet_stop.json", 'w') as f:
		json.dump(storedictstop.values(), f)
	with open("wordnet_duration.json", 'w') as f:
		json.dump(storedictduration.values(), f)

# print sum([len(e["words"]) for e in storedict])/float(len(storedict))

# df = pd.DataFrame(storedict.values()[1:], columns = ["start", "stop", "words"])
# df.to_csv("cleaned.csv")
