from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet
import gensim.downloader as api
import numpy as np
from util import Loader
from ldamallet import LdaMalletHandler
from sklearn.feature_extraction.text import CountVectorizer
import os

l = Loader()
d = l.from_text_line_by_line('sbrt_corpus_preproc.txt')

vect = CountVectorizer()
X = vect.fit_transform(d)

print(X.shape)

path_to_mallet_binary = "/home/thiagodepaulo/Mallet/bin/mallet"
model = LdaMalletHandler(n_components=100,mallet_path=path_to_mallet_binary,
iterations=2000, vectorizer=vect )

X_red = model.fit_transform(X)


def create_files(corpus, X, beta, gamma, vocab, docs, prefix):
    os.makedirs(prefix, exist_ok=True)

    with open(os.path.join(prefix, 'corpus.txt'),'wt') as fout:
        fout.write('\n'.join(corpus))

    from scipy import sparse
    X_coo = sparse.coo_matrix(X)
    l=[[i] for i in range(X.shape[0])]
    for i,j,v in zip(X_coo.row, X_coo.col, X_coo.data):
        #print("row = %d, column = %d, value = %s" % (i,j,v))
        l[i].append((j,v))

    s = ''
    for d in l:
        s += str(d[0])+' '
        s += ' '.join([ str(t[0])+':'+str(t[1]) for t in d[1:]])+'\n'

    with open(os.path.join(prefix, 'doc_wordcount_file.txt'),'wt') as fout:
        fout.write(s)

    with open(os.path.join(prefix, 'beta.txt'),'wt') as fout:
        np.savetxt(fout,beta)

    with open(os.path.join(prefix, 'gamma.txt'),'wt') as fout:
        np.savetxt(fout,gamma)

    with open(os.path.join(prefix, 'vocab_file.txt'),'wt') as fout:
        fout.write('\n'.join(vocab))

    with open(os.path.join(prefix, 'doc_file.txt'),'wt') as fout:
        fout.write('\n'.join(docs))

docs_names = ['doc_'+str(i) for i in range(X.shape[0])]
sorted_vocab = sorted(vect.vocabulary_.items(), key=lambda tup: tup[1])
vocab = [t[0] for t in sorted_vocab]
d_original = d = l.from_text_line_by_line('sbrt_corpus.txt')
create_files(d, X, model.get_topic_words_matrix(), model.get_doc_topic_matrix(), vocab, docs_names, 'sbrt')
