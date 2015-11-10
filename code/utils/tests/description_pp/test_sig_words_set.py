import code.utils.description.text_processing

stopwords = open("../../utils/description_pp/stopwords.txt").read().splitlines()

def test_significant_words_set():
	actual = significant_words_set("fake_movie_description.csv", stopwords)
	expected = set(["hi", "hello", "lol", "k"])
	assert actual == expected