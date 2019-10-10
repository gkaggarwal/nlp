import argparse
from nltk.tag import hmm
import os
import nltk
nltk.download('brown')
nltk.download('universal_tagset')  

from nltk.corpus import brown
from nltk import sent_tokenize, word_tokenize




def split_list_char(src):
	dest = []
	for sent in src:
		temp = []
		s_temp = sent_tokenize(sent)
		for word in sent:
			for c in word:
				if c != '\n':
					temp.append(c)
		dest.append(temp)
	return dest

def load_files(p='a2data/cipher1/', mode='train'):
	tagged_sentences = brown.tagged_sents(categories="news", tagset="universal")

	# let's keep 20% of the data for testing, and 80 for training
	i = int(len(tagged_sentences)*0.2)
	train_sentences = tagged_sentences[i:]
	test_sentences = tagged_sentences[:i]
	print(test_sentences[0])

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
	print(corprus[0])
	return corprus



def hmm_base():
	train_corpus = load_files()
	test_corpus = load_files(mode='test')
	trainer = hmm.HiddenMarkovModelTrainer()
	tagger = trainer.train_supervised(train_corpus)

	res = tagger.evaluate(test_corpus)

	# accruacy
	print(res)






def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lm', action='store_true')
	parser.add_argument('-hmm', action='store_true')
	args = parser.parse_args()

	if args.lm is False and args.hmm is False:
		hmm_base()



if __name__ == '__main__':
	main()