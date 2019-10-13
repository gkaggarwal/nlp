import argparse
from nltk.tag import hmm
import os
import nltk
from nltk.probability import LidstoneProbDist, MLEProbDist, FreqDist, ConditionalFreqDist, ConditionalProbDist
import string
nltk.download('brown')
nltk.download('universal_tagset')  

from nltk.corpus import brown
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

def split_list_char(src):
	dest = []
	for sent in src:
		temp = []
		for word in sent:
			for c in word:
				if c != '\n':
					temp.append(c)
		dest.append(temp)
	return dest


def load_files(p='a2data/cipher3/', mode='train'):
	tagged_sentences = brown.tagged_sents(categories="news", tagset="universal")

	with open((p+mode+'_plain.txt'), 'r') as f:
		label = f.readlines()

	with open((p+mode+'_cipher.txt'), 'r') as f:
		data = f.readlines()

	label_char = split_list_char(label)
	data_char = split_list_char(data)
	# print(label_char[0])
	corprus = []
	# merge data label

	for i in range(len(data_char)):
		temp = []
		for j in range(len(data_char[i])):
			#temp.append((data_char[i], label_char[i]))
			temp.append((data_char[i][j], label_char[i][j]))
		corprus.append(temp)
	return corprus



def hmm_base():
	train_corpus = load_files()
	test_corpus = load_files(mode='test')
	trainer = hmm.HiddenMarkovModelTrainer()
	tagger = trainer.train_supervised(train_corpus)
	print(test_corpus[0])
	res = tagger.evaluate(test_corpus)

	# accruacy
	print(res)


def hmm_laplace():
	train_corpus = load_files()
	test_corpus = load_files(mode='test')

	def est(fd, bins):
		return LidstoneProbDist(fd, 1, bins)	

	trainer = hmm.HiddenMarkovModelTrainer()
	tagger = trainer.train_supervised(train_corpus, estimator=est)
	tagger.train(train_corpus)
	# print(test_corpus[0])
	res = tagger.evaluate(test_corpus)

	# accruacy
	print(res)


def train_supervised_modified(labelled_sequences, extra_transition, estimator=None):
    """
    The following are taken directly from NLTK's website
    Since it is impossible to modify the A matrix by NLTK's design
    """

    # default to the MLE estimate
    _symbols = []
    _states = []
    if estimator is None:
        estimator = lambda fdist, bins: MLEProbDist(fdist)

    # count occurrences of starting states, transitions out of each state
    # and output symbols observed in each state
    known_symbols = set(_symbols)
    known_states = set(_states)

    starting = FreqDist()
    transitions = ConditionalFreqDist()
    outputs = ConditionalFreqDist()
    for sequence in labelled_sequences:
        lasts = None
        for token in sequence:
            state = token[0]
            symbol = token[1]
            if lasts is None:
                starting[state] += 1
            else:
                transitions[lasts][state] += 1
            outputs[state][symbol] += 1
            lasts = state

            # update the state and symbol lists
            if state not in known_states:
                _states.append(state)
                known_states.add(state)

            if symbol not in known_symbols:
                _symbols.append(symbol)
                known_symbols.add(symbol)

    # create probability distributions (with smoothing)
    N = len(_states)
    pi = estimator(starting, N)
    A = ConditionalProbDist(transitions.__add__(extra_transition), estimator, N)
    B = ConditionalProbDist(outputs, estimator, len(_symbols))

    return hmm.HiddenMarkovModelTagger(_symbols, _states, A, B, pi)


def extra_text_import():
	news_text = brown.words(categories='news')
	words = news_text
	table = str.maketrans('', '', string.punctuation)
	
	words = [word.lower() for word in words]
	print(words[:20])
	stripped = [w.translate(table) for w in words]
	print(stripped[0:20])


def extra_transition():
	"""Taken from  http://www.nltk.org/api/nltk.tag.html#nltk.tag.hmm.HiddenMarkovModelTrainer.train_supervised"""
	labelled_sequences = brown.words(categories='news')
	transitions = ConditionalFreqDist()
	for sequence in labelled_sequences:
		lasts = None
		for token in sequence:
			state = token[1]
			if lasts is not None:
				transitions[lasts][state] += 1
	return transitions
	

def hmm_extra():
	train_corpus = load_files()
	test_corpus = load_files(mode='test')
	extra_count = extra_transition()
	tagger = train_supervised_modified(train_corpus, extra_count)
	print(test_corpus[0])
	res = tagger.evaluate(test_corpus)
	print(res)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lm', action='store_true')
	parser.add_argument('-hmm', action='store_true')
	args = parser.parse_args()

	if args.lm is False and args.hmm is False:
		# hmm_base()
		# extra_trainsition()
		# hmm_extra()
		extra_text_import()


if __name__ == '__main__':
	main()