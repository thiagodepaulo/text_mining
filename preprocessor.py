
import re
import nltk
import string
import unicodedata
from tempfile import mkdtemp
from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import RSLPStemmer
from sklearn.externals.joblib import Memory
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
import inspect

class StopWords(TransformerMixin):
    """
    Stop-words extractor class

    ...

    Attributes
    ----------
    lang : str
        language of stop-word list. Portuguese by default
    stopword_list : list
         list of stopwords. It is defined by defaul by nltk.corpus.stopwords.words(lang), where lang is the language
    remove_accents : bool
        flag variable to indicate whether remove or not the accents of strings
    tokenize : bool
        flag variable to indicate whether tokenize or not the string. If it is True return the list of list of words,
        if it is False, return a list of strings

    Methods
    ---------
    fit(self, X, y=None)

    transform(self, X, *_)

    emove_stopwords(self, tokens)


    """

    def __init__(self, lang='portuguese', stopword_list=None, remove_accents=False, tokenize=False):
        self.lang = lang
        self.stopword_list = stopword_list
        self.tokenize = tokenize
        if self.stopword_list is None:
            self.stopword_list = nltk.corpus.stopwords.words(self.lang)
        self.remove_accents = remove_accents
        if self.remove_accents:
            new_stopword_list = []
            for word in self.stopword_list:
                word = unicodedata.normalize('NFKD', word).encode('ISO-8859-1', 'ignore')
                word = word.decode('ISO-8859-1')
                new_stopword_list.append(word)
            self.stopword_list = new_stopword_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        doc_list = []
        for doc in X:
            tokens = doc
            if isinstance(doc,str):
                tokens = nltk.word_tokenize(text=doc)
            doc_list.append(self.remove_stopwords(tokens))
        if not self.tokenize:
            return [ ' '.join(tokens) for tokens in doc_list]
        return doc_list

    def remove_stopwords(self, tokens):
        filtered_tokens = [token for token in tokens if token not in self.stopword_list]
        return filtered_tokens


class Cleaner(TransformerMixin, BaseEstimator):

    def __init__(self, remove_accents=True, remove_alpha_numeric=True, max_word_lenght=2):
        self.remove_accents = remove_accents
        self.remove_alpha_numeric = remove_alpha_numeric
        self.max_word_lenght = max_word_lenght

    def fit(self, X, y=None):
        return self

    def replace_blank(self, string):
        while '  ' in string:
            string = string.replace('  ', ' ')
        return string

    def transform(self, X, *_):
        punctuation_table = table = str.maketrans(dict.fromkeys(string.punctuation))
        text_list = []
        for text in X:
            text = text.lower()
            if self.remove_accents:
                text = unicodedata.normalize('NFKD', text).encode('ISO-8859-1', 'ignore')
                text = text.decode('ISO-8859-1')
            if self.remove_alpha_numeric:
                text = re.sub(r'[^a-z ]', r'', text)
            else:
                #do not remove alpha-numeric characteres, but remove punctuation from punctuation table
                text = text.translate(punctuation_table)
            # remove string with lenght up to self.max_word_lenght
            text = re.sub(r'\b\w{,%s}\b' %self.max_word_lenght, '', text)
            # remove double blank space
            text = self.replace_blank(text)
            text_list.append(text)
        return text_list

class Stemmer(TransformerMixin, BaseEstimator):

    def __init__(self, lang='portuguese', stemmer_obj=None, tokenize=False, fit_reuse = False):
        self.lang = lang
        self.stemmer_obj=stemmer_obj
        if stemmer_obj is None:
            self.stemmer_obj=SnowballStemmer(lang)
        self.tokenize=tokenize
        self.fit_reuse = fit_reuse

    def stemmer_obj_options(self, lang):
        if lang == 'portuguese':
            return [RSLPStemmer(), SnowballStemmer('portuguese')]
        elif lang=='english':
            return [PorterStemmer(),SnowballStemmer('english')]
        else:
            print('language not supported')
            return None

    def stem(self, word):
        if word in self.word_stem:
            word_stemmed = self.word_stem[word]
        else:
            word_stemmed = self.stemmer_obj.stem(word)
            self.word_stem[word] = word_stemmed
            if word_stemmed not in self.stem_word:
                self.stem_word[word_stemmed] = []
        self.stem_word[word_stemmed].append(word)
        return word_stemmed

    def stem_corpus(self, X):
        self.stem_word = {}
        self.word_stem = {}
        X_stemmed = []
        for tokens in X:
            if isinstance(tokens, str):
                tokens = nltk.word_tokenize(text=tokens)
            X_stemmed.append([ self.stem(word) for word in tokens])
        return X_stemmed

    def fit(self, X, y=None):
        self.X_stemmed = self.stem_corpus(X)
        return self

    def transform(self, X, *_):
        if self.fit_reuse:
            r = self.X_stemmed
        else:
            r = self.stem_corpus(X)
        if not self.tokenize:
            r = [' '.join(tokens) for tokens in r]
        return r

    def predict(self, X):
        if not hasattr(self, 'stem_counters'):
            self.stem_counters = {w: Counter(self.stem_word[w]) for w in self.stem_word}
        X_ = []
        for tokens in X:
            if isinstance(tokens, str):
                tokens = nltk.word_tokenize(text=tokens)
            X_.append([ self.stem_counters[word].most_common(1)[0][0]
                       if word in self.stem_counters else word for word in tokens])
        return X_


class Preprocessor(TransformerMixin):

    def __init__(self, cache=False, **kwargs):
        if cache:
            cachedir = mkdtemp()
            memory = Memory(cachedir=cachedir, verbose=10)
        self.pipe = Pipeline([
            ('stopwords',StopWords(**self.class_params(StopWords,kwargs))),
            ('stemming', Stemmer(**self.class_params(Stemmer,kwargs))),
            ('cleanning', Cleaner(**self.class_params(Cleaner,kwargs))),
        ])

    def class_params(self, Class, kwargs):
        args = inspect.getfullargspec(Class.__init__)[0][1:]
        defaults = inspect.getfullargspec(Class.__init__)[3]
        new_kwargs = {}
        for i, key in enumerate(args):
            if key not in kwargs:
                new_kwargs[key] = defaults[i]
            else:
                new_kwargs[key] = kwargs[key]
        return new_kwargs

    def fit(self, X, y=None):
        return self.pipe.fit(X,y)

    def transform(self, X, *_):
        return self.pipe.transform(X)
