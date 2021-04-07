import pandas as pd
import gensim
from os import listdir
from tqdm import tqdm 

HOME = 'datasets/itd_corpus'
def read_corpus(tokens_only=False):
    for i, f in tqdm(enumerate(listdir(HOME))):
        doc = open(HOME+'/'+f,'r').read()
        tokens = gensim.utils.simple_preprocess(doc)
        
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_corpus = list(read_corpus())
model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=5, epochs=10)
model.build_vocab(read_corpus())
model.train(read_corpus(), total_examples=model.corpus_count, epochs=model.epochs)
model.save('models/itd_doc2vec_model')
# model = gensim.models.doc2vec.Doc2Vec.load('models/itd_doc2vec_model')