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

# TODO:
# write down every config
# no pipeline for stem or lemma since it's binary anyways, just do it manually!!!

# data preprocessing imports

import nltk
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer


# sklearn model imports
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


# META DATA
POS_FILENAME = "data/rt-polaritydata/rt-polarity.pos"
NEG_FILENAME = "data/rt-polaritydata/rt-polarity.neg"


class Preprocess:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    @classmethod
    def stem(self, X):
        """
		dummy_y will not be used
		"""
        corporus = X
        result = []
        for x in corporus:
            words = word_tokenize(x)
            res = [PorterStemmer().stem(i) for i in words]
            result.append(" ".join(res))
        return result

    @classmethod
    def lemma(self, X):
        corporus = X
        result = []
        for x in corporus:
            words = word_tokenize(x)
            res = [WordNetLemmatizer().lemmatize(i) for i in words]
            result.append(" ".join(res))
        return result


# uniram features
class Unigram:
    def __init__(self):
        pass

    @classmethod
    def raw(self):
        # Implement raw unigram counts, nothing else fancy
        # default unigram config, we may change the parameters later

        return CountVectorizer()

    def params(self):
        params = {
            "clf__ngram_range": (1, 1),
            "clf__stop_words": [english, None],
            "min_df": [0.01, 0.05, 0.1],
        }
        return params

    @classmethod
    def rm_stopwords(self):
        # TODO param stop words, english or none
        return CountVectorizer(ngram_range=(1, 1), stop_words="english")

    @classmethod
    def rm_infreq_words(self):
        # TODO: different min df
        return CountVectorizer(ngram_range=(1, 1), min_df=0.01)

    def smoothing(self):
        pass


class Method:
    # not necessary
    def __init__(self):
        pass

    @classmethod
    def params(self, method):
        param_grid = None
        if method == "logistic":
            param_grid = [{"logistic__C": np.logspace(-4, 4, 20)}]
        elif method == 'svm':
            param_grid = [{'svm__C': np.logspace(-4, 4, 20)}]
        elif method == 'naive_bayes':
            param_grid = [{'naive_bayes__alpha':(0.1, 0.5, 1.0, 1.5, 2)}]
        return param_grid

    @classmethod
    def logistic(self):
        return LogisticRegression()

    @classmethod
    def svm(self):
        return SVC()

    @classmethod
    def naive_bayes(self):
        return BernoulliNB()

    @classmethod
    def dummy(self):
        return DummyClassifier()


# preprocessing
def load_to_text(file_name):
    """
	Load the .neg and .pos files
	Returns a type list
	"""
    with open(file_name, mode="r", encoding="cp1252") as jar:
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


def plot_confusion_mat(y_true, y_pred, name="cm"):

    labels = ["positive", "negative"]

    cm = confusion_matrix(y_pred, y_true)
    print("confusion_matrix")
    plt.figure()
    plt.matshow(cm)
    plt.savefig(name)


def get_results(clf, x_train, x_test, y_train, y_test, clf_name='q1'):

    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)

    print("train acc:", train_acc)
    print("test acc:", test_acc)
    print("best estimator", clf.best_estimator_)
    print("get_params", clf.best_params_)


def baseline(x_train, x_test, y_train, y_test):
    x_train = Preprocess.lemma(x_train)
    x_test = Preprocess.lemma(x_test)
   
    pipe = Pipeline(steps=[("raw", Unigram.raw()), ("logistic", Method.dummy())])
    pipe.fit(x_train, y_train)
    get_results(pipe, x_train, x_test, y_train, y_test)

    x_train = Preprocess.stem(x_train)
    x_test = Preprocess.stem(x_test)
   
    pipe = Pipeline(steps=[("raw", Unigram.raw()), ("logistic", Method.dummy())])
    pipe.fit(x_train, y_train)
    get_results(pipe, x_train, x_test, y_train, y_test)


def q1(x_train, x_test, y_train, y_test, unigram_meth=None):
    """
    with raw() and all the prediction methods
    """
    methods_to_try = ["svm"]#, "naive_bayes", "logistic", "dummy"]
    if unigram_meth is not None:
        unigram = unigram_meth
    else:
        unigram = Unigram.raw()
    for m in methods_to_try:
        if m == "svm":
            pipe = Pipeline(steps=[("raw", unigram), ("svm", Method.svm())])
            param = Method.params(m)
        elif m == "naive_bayes":
            pipe = Pipeline(steps=[("raw", unigram), ("naive_bayes", Method.naive_bayes())])
            param = Method.params(m)
        elif m == 'logistic':
            pipe = Pipeline(steps=[("raw", unigram), ("logistic", Method.logistic())])
            param = Method.params(m)
        clf = GridSearchCV(pipe, param_grid=param)
        clf.fit(x_train, y_train)
        get_results(clf, x_train, x_test, y_train, y_test)





def main():
    negative_reviews = load_to_text(NEG_FILENAME)
    positive_reviews = load_to_text(POS_FILENAME)

    # convert to unigram counts
    x, y = label_and_merge(positive_reviews, negative_reviews)
    preprocess = ["raw", "stem", "lemma"]
    methods_to_try = ["svm", "naive_bayes", "logistic", "dummy"]
    x_train, x_test, y_train, y_test = train_test_split(x, y)


    # for p in preprocess:
    #     if preprocess == "stem":
    #         x_train = Preprocess.stem(x_train)
    #         x_test = Preprocess.stem(x_test)
    #     elif preprocess == "lemma":
    #         x_train = Preprocess.lemma(x_train)
    #         x_test = Preprocess.lemma(x_test)


    q1(x_train, x_test, y_train, y_test)
    # print("experiments with raw unigrams")
    # q1(x_train, x_test, y_train, y_test)
    # print("unigram without infrequent words")
    # q1(x_train, x_test, y_train, y_test, Unigram.rm_infreq_words())
    # print("unigram remove stop words")
    # q1(x_train, x_test, y_train, y_test, Unigram.rm_stopwords())


if __name__ == "__main__":
    main()
