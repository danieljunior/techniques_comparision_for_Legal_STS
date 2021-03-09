import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('floresta')
nltk.download('rslp')
from nltk.corpus import stopwords
from NLPyPort.FullPipeline import *
import numpy as np

def bm25_tokenizer(corpus):
    nlpyport_options = {
            "tokenizer" : True,
            "pos_tagger" : True,
            "lemmatizer" : True,
            "entity_recognition" : False,
            "np_chunking" : False,
            "pre_load" : False,
            "string_or_array" : True
        }
    
    documents = []
    
    for document in corpus:
        doc = new_full_pipe(document, options=nlpyport_options)
        tokens = [lema for idx, lema in enumerate(doc.lemas)
                        if lema != 'EOS'
                        and lema != '']
        tokens = [token for token in tokens 
                  if token not in nltk.corpus.stopwords.words('portuguese')]
        
        tokens = [token for token in tokens 
                  if not re.match('[^A-Za-z0-9]+', token)]
        
        tokens = [token for token in tokens 
                  if any(char.isdigit() for char in token)]
        
        documents.append(tokens)
    return documents

def biggests_index(a,N): 
    return np.argsort(a)[::-1][:N]