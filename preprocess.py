"""
To preprocess the following:

- [ ] unigram counts
- [ ] lemmatize
- [ ] stem
- [ ] stopword
- [ ] infrequent words
- [ ] smoothing
- [ ] regularization
"""
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.model_selection import train_test_split


# META DATA
POS_FILENAME = 'data/rt-polaritydata/rt-polarity.pos'
NEG_FILENAME = 'data/rt-polaritydata/rt-polarity.neg'


def load_to_text(file_name):
	"""
	Load the .neg and .pos files
	Returns a type list
	"""
	with open(file_name, mode='r', encoding='cp1252') as jar:
		reviews = jar.readlines()
		jar.close()
	train_data, test_data = train_test_split(reviews)
	train_data = ''.join(train_data)
	test_data = ''.join(test_data)
	return train_data, test_data


def convert_unigram_raw(text):
	# Implement raw unigram counts, nothing else fancy
	token = nltk.word_tokenize(text)
	unigrams = ngrams(token, 1)
	return unigrams


def main():
	negative_reviews_train, negative_reviews_test = load_to_text(NEG_FILENAME)
	positive_reviews_train, positive_reviews_test = load_to_text(POS_FILENAME)
	print(Counter(convert_unigram_raw(negative_reviews_test)))




if __name__ == '__main__':
	main()



