# data file
description = open("../description_pp/description.csv") 
csv_description = csv.reader(description) 

# stopwords file
stopwords = open("../description_pp/stopwords.txt")
stopwords_list = stopwords.read().splitlines()

