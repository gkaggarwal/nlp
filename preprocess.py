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
	def __init__(self, data):
		self.text = data

	def raw(self):
		# Implement raw unigram counts, nothing else fancy
		# default unigram config, we may change the parameters later
		return CountVectorizer(ngram_range=(1, 1))

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


	label_pos = np.ones(len(data_pos))
	label_neg = np.zeros(len(data_neg))

	data = data_pos + data_neg
	label = np.concatenate([label_pos, label_neg])
	return data, label



def main():
	negative_reviews = load_to_text(NEG_FILENAME)
	positive_reviews = load_to_text(POS_FILENAME)

	# convert to unigram counts



	x, y = label_and_merge(positive_reviews, negative_reviews)
	n_grams = CountVectorizer(ngram_range=(1, 1))
	print(type(x))

	clf = LogisticRegression()
	clf1 = DummyClassifier()
	x_train, x_test, y_train, y_test = train_test_split(n_grams.fit_transform(x), y)

	clf.fit(x_train, y_train)
	print(clf.score(x_test, y_test))
	
	clf1.fit(x_train, y_train)
	print(clf1.score(x_test, y_test))

if __name__ == '__main__':
	main()
