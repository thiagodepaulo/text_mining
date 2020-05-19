from sklearn.base import BaseEstimator,TransformerMixin
from gensim.matutils import Sparse2Corpus, corpus2dense
from gensim.models.wrappers import LdaMallet
from gensim.corpora.dictionary import Dictionary
import numpy as np


class LdaMalletHandler(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=100, mallet_path=None, prefix=None, iterations=1000, vectorizer=None ):
        self.n_components=n_components
        self.mallet_path=mallet_path
        self.prefix=prefix
        self.iterations=iterations
        self.vectorizer = vectorizer

    def vect2gensim(self, vectorizer, dtmatrix):
        # transform sparse matrix into gensim corpus and dictionary
        corpus_vect_gensim = Sparse2Corpus(dtmatrix, documents_columns=False)
        dictionary = Dictionary.from_corpus(corpus_vect_gensim,
                id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
        return (corpus_vect_gensim, dictionary)

    def fit(self, X, y=None):
        print('vect2gensim')
        corpus, dictionary = self.vect2gensim(self.vectorizer,X)
        self.model = LdaMallet(self.mallet_path, iterations=self.iterations, corpus=corpus, num_topics=self.n_components, id2word=dictionary)
        return self

    def transform(self, X):
        corpus = Sparse2Corpus(X, documents_columns=False)
        doc_topic = self.model[corpus]
        mat = np.zeros((X.shape[0], self.n_components), dtype=np.float64)
        for did, doc in enumerate(doc_topic):
            for topic in doc:
                mat[did][topic[0]] = topic[1]
        return mat

    def get_doc_topic_matrix(self):
        arr = []
        lines = open(self.model.fdoctopics(), "r").read().splitlines()
        for line in lines:
            arr.append(line.split()[2:])
        return np.array(arr, dtype=np.float64)

    def get_topic_words_matrix(self):
        return self.model.get_topics()
