import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('floresta')
nltk.download('rslp')
from nltk.corpus import stopwords
from NLPyPort.FullPipeline import *
import numpy as np
import ray
from tqdm import tqdm
from annoy import AnnoyIndex
from rank_bm25 import BM25Plus

# from longformer_embedder import LongformerEmbedder
# from bert_embedder import BertEmbedder
# from word2vec_embedder import Word2VecEmbedder
# from tfidf_embedder import TfIdfEmbedder
# from elmo_embedder import ElmoEmbedder
from sentence_transformer_embedder import SentenceTransformerEmbedder
from transformers_embedder import TransformersEmbedder
# from doc2vec_embedder import Doc2VecEmbedder
# from lda_embedder import LDAEmbedder

# ray.init(num_cpus=4, ignore_reinit_error=True)

def nlpyport_tokenizer(corpus):
    documents = [tokenize(document) for document in corpus]

    return documents

def textos_preprocessados(docs):
    ray.init(num_cpus=4, ignore_reinit_error=True)

    results = []
    for texto in tqdm(docs):
        results.append(tokenize.remote(texto))

    preprocessado = []
    for i in results:
        d = ray.get(i)
        preprocessado.append(' '.join(d))

    return preprocessado

@ray.remote
def tokenize(document):
    nlpyport_options = {
            "tokenizer" : True,
            "pos_tagger" : True,
            "lemmatizer" : True,
            "entity_recognition" : False,
            "np_chunking" : False,
            "pre_load" : False,
            "string_or_array" : True
        }
    doc = new_full_pipe(document, options=nlpyport_options)
    tokens = [lema for idx, lema in enumerate(doc.lemas)
                    if lema != 'EOS'
                    and lema != '']
    tokens = [token for token in tokens
              if token not in nltk.corpus.stopwords.words('portuguese')]

    tokens = [token for token in tokens
              if not re.match('[^A-Za-z0-9]+', token)]

    tokens = [token for token in tokens
              if not any(char.isdigit() for char in token)]

    return tokens

def biggests_index(a,N): 
    return np.argsort(a)[::-1][:N]

def setup_indexer(vectors_size=3072):
    return AnnoyIndex(vectors_size, 'angular')

### Load models
def tfidf(data):    
    tfidf = TfIdfEmbedder(data)
    tfidf.set_indexer(setup_indexer(vectors_size=len(tfidf.get_feature_names())))
    return tfidf

def word2vec():
    return Word2VecEmbedder('models/word2vec/skip_s300.txt', 
                            indexer = setup_indexer(vectors_size=300))

def weighted_word2vec(tfidf, dictionary):
    return Word2VecEmbedder('models/word2vec/skip_s300.txt', 
                            tfidf_model=tfidf, 
                            tfidf_dictionary=dictionary,
                            indexer = setup_indexer(vectors_size=300))

def fasttext():
    return Word2VecEmbedder('models/fasttext/skip_s300.txt', 
                            indexer = setup_indexer(vectors_size=300))
    
def weighted_fasttext(tfidf, dictionary):
    return Word2VecEmbedder('models/fasttext/skip_s300.txt', 
                            tfidf_model=tfidf, 
                            tfidf_dictionary=dictionary,
                            indexer = setup_indexer(vectors_size=300))
def lda(data, num_topics=44):
    return LDAEmbedder(data, 
                       indexer=setup_indexer(vectors_size=num_topics), 
                       num_topics=num_topics)
    
def doc2vec():
    return Doc2VecEmbedder('models/itd_doc2vec_model',
                           indexer = setup_indexer(vectors_size=100))

def sentence_transformer():
    return SentenceTransformerEmbedder('models/portuguese_sentence_transformer', 
                                        setup_indexer(vectors_size=768))

def sim_cse():
    return SentenceTransformerEmbedder('models/portuguese_sim_cse',
                                        setup_indexer(vectors_size=768))

def diff_cse():
    return TransformersEmbedder('models/portuguese_diff_cse',
                                        setup_indexer(vectors_size=768))

def elmo():
    options_path = 'models/elmo/options.json'
    weights_path = 'models/elmo/elmo_pt_weights_dgx1.hdf5'
    return ElmoEmbedder(options_path, weights_path, setup_indexer(vectors_size=1024))

def bert():
    return BertEmbedder('models/bert-base-cased-pt-br', setup_indexer())

def bertikal():
    return BertEmbedder('models/BERTikal', setup_indexer())

def itd_bert():
    return BertEmbedder('models/itd_bert', setup_indexer())

def longformer(): 
    return LongformerEmbedder('models/bert_longformer', setup_indexer())

def itd_longformer():
    return LongformerEmbedder('models/itd_bert_longformer', setup_indexer())

def bm25(data):
    return BM25Plus(data)

def embedders():
    return {
        # 'tfidf': tfidf,
        # 'bm25': bm25,
        # 'lda': lda,
        # 'word2vec': word2vec,
        # 'weighted_word2vec': weighted_word2vec,
        # 'fasttext': fasttext,
        # 'weighted_fasttext': weighted_fasttext,
        # 'doc2vec': doc2vec,
        # 'sentence_transformer': sentence_transformer,
        'sim_cse': sim_cse,
        # 'diff_cse': diff_cse,
        # 'bert': bert,
        # 'bertikal': bertikal,
        # 'itd_bert': itd_bert,
        # 'longformer':longformer,
        # 'itd_longformer': itd_longformer,
        # 'elmo': elmo,
    }