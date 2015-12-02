import csv

def significant_words_set(csv_loc, stopwords):
	"""
	Return set of all significant words in a movie feature csv file
	"""
	with open(csv_loc) as f:
		c = csv.reader(f)
	uwords = set()
	for row in csv_description:
		r = row[3].lower().split()
		r = [re.sub(r'\W+', '', w) for w in r if re.sub(r'\W+', '', w) != ""]
		r = [w for w in r if w not in stopwords_list]
		uwords |= set(r)
	print uwords