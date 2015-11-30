# data file
description = open("description.csv") 
csv_description = csv.reader(description) 

# stopwords file
stopwords = open("stopwords.txt")
stopwords_list = stopwords.read().splitlines()

