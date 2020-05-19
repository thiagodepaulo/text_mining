import util
from preprocessor import Preprocessor

print('carregando corpus...')
l = util.Loader()
corpus = l.from_files_2('/exp/mctic/sbrt_txts/respostas/*', encod="ISO-8859-1")
print(corpus[1])
print(len(corpus))
print('fim carregamento.')
print('salvando corpus')
with open('sbrt_corpus.txt','wt') as fout:
    for d in corpus:
        fout.write(d.replace('\r',' ').replace('\n',' ')+'\n')
print('fim')

print('preporcessando...')
preproc = Preprocessor()
corpus_p = preproc.fit_transform(corpus)
print('fim preprocessamento')

print(corpus_p[1])
print(len(corpus_p))
print('salvando corpus preprocessado...')
with open('sbrt_corpus_preproc.txt','wt') as fout:
    for d in corpus_p:
        fout.write(d+'\n')
print('fim')
