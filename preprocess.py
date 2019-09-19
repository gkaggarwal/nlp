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
from sklearn.pipeline import Pipeline
# META DATA
POS_FILENAME = 'data/rt-polaritydata/rt-polarity.pos'
NEG_FILENAME = 'data/rt-polaritydata/rt-polarity.neg'


# uniram features
class Unigram:
	def __init__(self):
		pass

	@classmethod
	def raw(self, corporus):
		# Implement raw unigram counts, nothing else fancy
		# default unigram config, we may change the parameters later
		return CountVectorizer(ngram_range=(1, 1)).fit_transform(corporus)

	def lemmatize(self, x):
		pass
		
	@classmethod
	def stem(self, corporus):
		result = []
		for x in corporus:
			words = word_tokenize(x)
			res = [PorterStemmer().stem(i) for i in words]
			result.append(' '.join(res))
		return CountVectorizer(ngram_range=(1, 1)).fit_transform(result)
		

	@classmethod
	def rm_stopwords(self, corporus):
		return CountVectorizer(ngram_range=(1, 1), stop_words='english').fit_transform(corporus)

	@classmethod
	def rm_infreq_words(self, corporus): 
		return CountVectorizer(ngram_range=(1, 1), min_df=0.01).fit_transform(corporus)

	def smoothing(self):
		pass


class Method:
	# not necessary
	def __init__(self):
		pass

	@classmethod
	def logistic(self):
		return LogisticRegression()

	def svm(self):
		pass

	def naives_bayes(self):
		pass

	@classmethod
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


	# convert to unigram counts
	x, y = label_and_merge(positive_reviews, negative_reviews)

	x_train, x_test, y_train, y_test = train_test_split((Unigram.stem(x)), y)

	pipe = Pipeline(steps=[('logistic', Method.logistic())])

	pipe.fit(x_train, y_train)
	print(pipe.score(x_test, y_test))


if __name__ == '__main__':
	main()
