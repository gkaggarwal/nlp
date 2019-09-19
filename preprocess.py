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

# data preprocessing imports

from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 

# sklearn model imports
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# META DATA
POS_FILENAME = 'data/rt-polaritydata/rt-polarity.pos'
NEG_FILENAME = 'data/rt-polaritydata/rt-polarity.neg'


# uniram features
class Unigram:
	def __init__(self):
		pass

	@classmethod
	def raw(self):
		# Implement raw unigram counts, nothing else fancy
		# default unigram config, we may change the parameters later
		return CountVectorizer(ngram_range=(1, 1))

	def lemmatize(self, x):
		pass

	@classmethod
	def stem(self, corporus):
		result = []
		for x in corporus:
			words = word_tokenize(x)
			res = [PorterStemmer().stem(i) for i in words]
			result.append(' '.join(res))
		return result

	@classmethod
	def rm_stopwords(self):
		return CountVectorizer(ngram_range=(1, 1), stop_words='english')

	@classmethod
	def rm_infreq_words(self): 
		return CountVectorizer(ngram_range=(1, 1), min_df=0.01)

	def smoothing(self):
		pass


class Method:
	# not necessary
	def __init__(self):
		pass

	def logistic(self):
		return LogisticRegression()

	def svm(self):
		pass

	def naives_bayes(self):
		pass

	def dummy(self):
		return DummyClassifier()

# preprocessing
def load_to_text(file_name):
	"""
	Load the .neg and .pos files
	Returns a type list
	"""
	with open(file_name, mode='r', encoding='cp1252') as jar:
		reviews = jar.readlines()
		jar.close()
	# reviews = ''.join(reviews)

	return reviews


def label_and_merge(data_pos, data_neg):


	label_pos = np.ones(len(data_pos))
	label_neg = np.zeros(len(data_neg))

	data = data_pos + data_neg
	label = np.concatenate([label_pos, label_neg])
	return data, label



def main():
	negative_reviews = load_to_text(NEG_FILENAME)
	positive_reviews = load_to_text(POS_FILENAME)

	methods = [DummyClassifier(), LogisticRegression()]
	unigrams = [CountVectorizer(ngram_range=(1, 1)),
				CountVectorizer(ngram_range=(1, 1), stop_words='english'),
				CountVectorizer(ngram_range=(1, 1), min_df=0.01)]
	# convert to unigram counts
	x, y = label_and_merge(positive_reviews, negative_reviews)
	# may or may not stem


	n_grams = unigrams[2]
	stemmed = Unigram.stem(x)
	print('before', len(x))
	print("stemmed", len(stemmed))
	x_train, x_test, y_train, y_test = train_test_split(n_grams.fit_transform(stemmed), y)

	for clf in methods:
		print('method name: ', clf)
		clf.fit(x_train, y_train)
		print('test_accuracy:', clf.score(x_test, y_test))


if __name__ == '__main__':
	main()
