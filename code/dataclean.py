# 10/29/15 
# Cindy Mo  
# clean data from translated text file 
# remove stop words 

from collections import OrderedDict
import csv 
import numpy as np 
import pandas as pd 


description = open("description.csv") 
csv_description = csv.reader(description) 

storedict = OrderedDict()  

for row in csv_description: 
	key = (row[0],row[1]) 
	storedict[key] = row[3].lower() 	
	#print("start: " + row[0] + " end: " + row[1]) 
	#print(row[3]) 
# print(storedict)  
stopwords = open("stopwords.txt")
stopwords_list = stopwords.read().splitlines()
print(stopwords_list)

# remove stopwords 
storedict_final = OrderedDict()
for key in storedict.keys():
        text = storedict[key]
        text_list = text.split()
        # print(text_list)  
        new_array = [x.replace(".", "") for x in text_list] 
	new_array = [x for x in new_array if x not in stopwords_list] # removes stopwords
        # print(new_array) 
        # print(new_array) 
        new_array = [x for x in new_array if x not in ["a", "...", "s", "ss"]]
        new_array = [x.replace(":", "") for x in new_array] 
        new_array = [x.replace("...", "") for x in new_array] 
        new_array = [x.replace(";", "") for x in new_array] 
        storedict_final[key] =  new_array
print(storedict_final)
df = pd.DataFrame(storedict_final.items(), columns = ["startstop", "words"])
print(df)  
df.to_csv("cleaned.csv")
