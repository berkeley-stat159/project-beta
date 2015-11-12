from __future__ import absolute_import
import os, sys

files = [f for f in os.listdir('.') if os.path.isfile(f)]
print files

start_path = os.getcwd()
print start_path
# os.chdir('..')
# sys.path.append(os.path.abspath(os.getcwd()))
os.chdir('../../description_pp')
curr_path = os.path.abspath(os.getcwd())
sys.path.append(curr_path)
# print sys.path
os.chdir(start_path)

for d in sys.path:
	print "---------------"
	print d
	try:
		for f in os.listdir(d):
			if os.path.isfile(f):
				print f
	except:
		pass

files = [f for d in sys.path for f in os.listdir(d) if os.path.isfile(f) ]

import text_processing as tp

stopwords = open("stopwords.txt").read().splitlines()

def test_significant_words_set():
	actual = tp.significant_words_set("fake_movie_description.csv", stopwords)
	expected = set(["hi", "hello", "lol", "k"])
	assert actual == expected

print test_significant_words_set()