import re
import sklearn
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
            tokens = doc.split()
            return self._word_ngrams(tokens)
        return analyser
    
    def get_embeddings(self, text):
        return self.transform([text])
    
    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings.toarray()[0])
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)
    
    def __getstate__(self):
        return dict((k, v) for (k, v) in self.__dict__.items() if k != 'indexer')
