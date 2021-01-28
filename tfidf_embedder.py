import re
import sklearn
import spacy
from spacy.tokens import Doc
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('floresta')
nltk.download('rslp')
from nltk.corpus import stopwords
from NLPyPort.FullPipeline import *

nlp_global = spacy.load('pt_core_news_lg', 
                              disable=["parser","ner", "entity_linker", "textcat", 
                                       "entity_ruler"])
stemmer_global = nltk.stem.RSLPStemmer()

def lematizar(documento):
    nlpyport_options = {
            "tokenizer" : True,
            "pos_tagger" : True,
            "lemmatizer" : True,
            "entity_recognition" : False,
            "np_chunking" : False,
            "pre_load" : False,
            "string_or_array" : True
        }
    
    doc = new_full_pipe(documento, options=nlpyport_options)
    tokens = [lema for idx, lema in enumerate(doc.lemas)
                    if lema != 'EOS'
                    and lema != ''] # remover caracter de fim de linha
    return Doc(nlp_global.vocab, tokens)

def stemming(documento):
    tokens = [stemmer_global.stem(w) for w in documento]
    return Doc(nlp_global.vocab, tokens)

class VectorizerOptions(object):

    def __init__(self, options):
        self.options = options
        self.nlp = spacy.load('pt_core_news_lg', 
                              disable=["parser","ner", "entity_linker", "textcat", 
                                       "entity_ruler"])
        self.stopwords = nltk.corpus.stopwords.words('portuguese')
    
    def run(self, doc):        
        if self.check_preprocessor_option('lemmatization'):
            self.nlp.tokenizer = lematizar
        
        if self.check_preprocessor_option('stemming'):
            self.nlp.tokenizer = stemming
            
        doc_ = self.nlp(doc)
        return [token.text 
                for token in doc_
                if self.nao_remover_token(token)]
    
    def nao_remover_token(self, token):
        #remove nomes próprios
        if token.pos_ == 'PROPN':
            return False
        
        if self.check_preprocessor_option('stopwords') \
            and token.text in self.stopwords:
            return False

        if self.check_preprocessor_option('specials') \
            and re.match('[^A-Za-z0-9]+', token.text):
            return False
        
        if self.check_preprocessor_option('numbers') \
            and any(char.isdigit() for char in token.text):
            return False
        
        return True

    def check_preprocessor_option(self, option):
        return (self.options.get(option) and
                self.options[option])

class TfIdfEmbedder(sklearn.feature_extraction.text.TfidfVectorizer):

    def __init__(self, data, options={'stopwords': True, 'lemmatization': True,
                                'specials': True, 'numbers': True, }, *args,
                 **kwargs):
        self.options = options
        super(TfIdfEmbedder, self).__init__(*args, **kwargs)
        self.fit(data)
    
    def set_indexer(self, indexer):
        self.indexer = indexer

    def build_analyzer(self):
        def analyser(doc):
            tokens = VectorizerOptions(self.options).run(doc)
            return self._word_ngrams(tokens)
        return analyser
    
    def get_embeddings(self, text):
        return self.transform([text])
    
    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings.toarray()[0])
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)