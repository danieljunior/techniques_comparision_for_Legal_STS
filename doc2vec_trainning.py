import pandas as pd
import gensim
from os import listdir
from tqdm import tqdm 
            
class ITDIterator():
    
    HOME = 'datasets/itd_corpus'
    
    def __init__(self):
        self.generator_function = self.read_corpus
        self.generator = self.generator_function()
    
    def read_corpus(self, tokens_only=False):
        for i, f in tqdm(enumerate(listdir(ITDIterator.HOME))):
            doc = open(ITDIterator.HOME+'/'+f,'r').read()
            tokens = gensim.utils.simple_preprocess(doc)

            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result


# train_corpus = list(read_corpus())
model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=5, epochs=10)
model.build_vocab(ITDIterator())
model.train(ITDIterator(), total_examples=model.corpus_count, epochs=model.epochs)
model.save('models/itd_doc2vec_model')
# model = gensim.models.doc2vec.Doc2Vec.load('models/itd_doc2vec_model')