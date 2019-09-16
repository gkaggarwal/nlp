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
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.model_selection import train_test_split
# sklearn model imports
from sklearn.dummy import DummyClassifier
import numpy as np

# META DATA
POS_FILENAME = 'data/rt-polaritydata/rt-polarity.pos'
NEG_FILENAME = 'data/rt-polaritydata/rt-polarity.neg'


# uniram features
class Unigram:
	def __init__(self, data):
		self.text = data

	def raw(self):
		# Implement raw unigram counts, nothing else fancy
		token = nltk.word_tokenize(self.text)
		unigrams = ngrams(token, 1)
		return unigrams

	def lemmatize(self):
		pass

	def stem(self):
		pass

	def rm_stopwords(self):
		pass

	def rm_infreq_words(self): 
		pass

	def smoothing(self):
		pass


class Method:
	# not necessary
	def __init__(self):
		pass

	def logistic(self):
		pass

	def svm(self):
		pass

	def naives_bayes(self):
		pass




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
	data_pos = ''.join(data_pos)
	data_neg = ''.join(data_neg)


	label_pos = np.ones(len(data_pos))
	label_neg = np.zeros(len(data_neg))
	data = data_pos + data_neg
	label = np.concatenate([label_pos, label_neg])
	return data, label




def main():
	negative_reviews = load_to_text(NEG_FILENAME)
	positive_reviews = load_to_text(POS_FILENAME)
	x, y = label_and_merge(positive_reviews, negative_reviews)

	x_train, x_test, y_train, y_test = train_test_split(x, y)



	clf = DummyClassifier(strategy='most_frequent', random_state=0)
	clf.fit(x_train, y_train)
	print(clf.score(x_test, y_test))


if __name__ == '__main__':
	main()
