import sys
import code.utils.description_pp.text_processing as tp

# data file
try:
	description = sys.argv[1]
except:
	description = "description.csv" 


# stopwords file
stopwords = open("stopwords.txt")
stopwords_list = stopwords.read().splitlines()

sig_words = tp.significant_words_set(description, stopwords_list)

print sig_words