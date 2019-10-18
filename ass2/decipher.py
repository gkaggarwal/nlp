import argparse
from nltk.tag import hmm
import os
import nltk
from nltk.probability import (
    LidstoneProbDist,
    MLEProbDist,
    FreqDist,
    ConditionalFreqDist,
    ConditionalProbDist,
    LaplaceProbDist
)
import string

nltk.download("brown")
nltk.download("universal_tagset")

from nltk.corpus import brown
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer


def split_list_char(src):
    dest = []
    for sent in src:
        temp = []
        for word in sent:
            for c in word:
                if c != "\n":
                    temp.append(c)
        dest.append(temp)
    return dest


def load_files(p="a2data/cipher1/", mode="train"):

    with open(os.path.join(p, mode + "_plain.txt"), "r") as f:
        label = f.readlines()

    with open(os.path.join(p, mode + "_cipher.txt"), "r") as f:
        data = f.readlines()

    label_char = split_list_char(label)
    data_char = split_list_char(data)
    # print(label_char[0])
    corprus = []
    # merge data label

    for i in range(len(data_char)):
        temp = []
        for j in range(len(data_char[i])):
            # temp.append((data_char[i], label_char[i]))
            temp.append((data_char[i][j], label_char[i][j]))
        corprus.append(temp)
    return corprus


def hmm_base(path):
    train_corpus = load_files(p=path)
    test_corpus = load_files(p=path, mode="test")
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_corpus)
    res = tagger.evaluate(test_corpus)

    # accruacy
    print(res)


def hmm_laplace(path):
    train_corpus = load_files(p=path)
    test_corpus = load_files(p=path, mode="test")

    def est(fd, bins):
        return LidstoneProbDist(fd, 1, bins)

    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_corpus, estimator=est)
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
    if estimator is None:
        estimator = lambda fdist, bins: MLEProbDist(fdist)

    # count occurrences of starting states, transitions out of each state
    # and output symbols observed in each state
    known_symbols = []
    known_states = []

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
                known_states.append(state)

            if symbol not in known_symbols:
                known_symbols.append(symbol)

    extra_sequences = extra_text_import()
    for sequence in extra_sequences:
        lasts = None
        for token in sequence:
            state = token
            if lasts is None:
                starting[state] += 1
            else:
                transitions[lasts][state] += 1
            lasts = state

            # update the state and symbol lists
            if state not in known_states:
                known_states.append(state)



    # create probability distributions (with smoothing)
    N = len(known_states)
    # print("known_states", known_states)
    # print("len known")
    # print(N)
    pi = estimator(starting, N)
    A = ConditionalProbDist(transitions, estimator, N)
    B = ConditionalProbDist(outputs, estimator, len(known_symbols))

    return hmm.HiddenMarkovModelTagger(known_states, known_symbols, A, B, pi)


def extra_text_import():

    news_text = brown.sents(categories="news")
    words = []
    for sent in news_text:
        temp = sent
        words.append(TreebankWordDetokenizer().detokenize(temp))

    remove = string.punctuation + string.digits
    remove = remove.replace(",", "")
    remove = remove.replace(".", "")
    # print("patterns to remove", remove)
    table = str.maketrans("", "", remove)

    words = [w.lower() for w in words]
    words = [w.translate(table) for w in words]

    return words #[0:1]


def extra_transition():
    """Taken from  
    http://www.nltk.org/api/nltk.tag.html#nltk.
    tag.hmm.HiddenMarkovModelTrainer.train_supervised"""

    sentences = extra_text_import()
    transitions = ConditionalFreqDist()
    for sent in sentences:
        lasts = None
        for token in sent:
            #print(token)
            if lasts is None:
                pass
            else:
                transitions[lasts][token] += 1
            lasts = token

    return transitions


def hmm_extra(path):
    train_corpus = load_files(p=path)
    test_corpus = load_files(p=path, mode="test")
    extra_count = extra_transition()
    tagger = train_supervised_modified(train_corpus, extra_count)
    # tagger.train(train_corpus)
    res = tagger.evaluate(test_corpus)
    print(res)
    res = tagger.evaluate(train_corpus)
    print('train {}'.format(res))



def hmm_extra_laplace(path):
    train_corpus = load_files(p=path)
    test_corpus = load_files(p=path, mode="test")
    extra_count = extra_transition()

    def est(fd, bins):
        # if bins < fd.B():
        #     bins = fd.B()
        return LidstoneProbDist(fd, 1, bins)

    tagger = train_supervised_modified(train_corpus, extra_count, estimator=est)
    res = tagger.evaluate(test_corpus)
    print(res) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lm", action="store_true")
    parser.add_argument("-laplace", action="store_true")
    parser.add_argument('path')
    args = parser.parse_args()
    
    if args.lm is False and args.laplace is False:
        for i in range(3):
            real_num = i + 1
            fp = args.path + '/cipher' + str(real_num)
            print("Running {} on cipher {}".format("hmm_base", real_num))
            hmm_base(fp)
    elif args.lm is False and args.laplace is True:
        for i in range(3):
            real_num = i + 1
            fp = args.path + '/cipher' + str(real_num)
            print("Running {} on cipher {}".format("hmm_laplace", real_num))
            hmm_laplace(fp)
    elif args.lm is True and args.laplace is False:
        for i in range(3):
            real_num = i + 1
            fp = args.path + '/cipher' + str(real_num)
            print("Running {} on cipher {}".format("hmm_lm", real_num))
            hmm_extra(fp)
    elif args.lm is True and args.laplace is True:
        for i in range(3):
            real_num = i + 1
            fp = args.path + '/cipher' + str(real_num)
            print("Running {} on cipher {}".format("hmm_lm_laplace", real_num))
            hmm_extra_laplace(fp)


if __name__ == "__main__":
    main()
