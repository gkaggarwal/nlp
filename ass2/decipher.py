import argparse
from nltk.tag import hmm
import os
import nltk
nltk.download('treebank')


def split_list_char(src):
	dest_char = []

	# break down a list into charactors
	for i in range(len(src)):
		
		dest_char.append(src[i].split())

	return dest_char


def load_files(p='a2data/cipher1/', mode='train'):
	with open((p+mode+'_plain.txt'), 'r') as f:
		label = f.readlines()

	with open((p+mode+'_cipher.txt'), 'r') as f:
		data = f.readlines()

	label_char = split_list_char(label)
	data_char = split_list_char(data)
	print(data_char[0])
	corprus = []
	# merge data label
	for i in range(len(data_char)):
		corprus.append(((data_char[i], label_char[i])))

	return corprus



def hmm_base():
	train_corpus = load_files()
	test_corpus = load_files(mode='test')
	trainer = hmm.HiddenMarkovModelTrainer()
	tagger = trainer.train_supervised(train_corpus)
	print(tagger.tag(test_corpus))





def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lm', action='store_true')
	parser.add_argument('-hmm', action='store_true')
	args = parser.parse_args()

	if args.lm is False and args.hmm is False:
		hmm_base()



if __name__ == '__main__':
	main()