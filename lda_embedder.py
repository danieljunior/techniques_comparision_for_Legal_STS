import gensim
import gensim.corpora as corpora

class LDAEmbedder():
    def __init__(self, data, indexer, num_topics=50):
        self.id2word = corpora.Dictionary([t.split() for t in data])
        corpus = [self.id2word.doc2bow(text.split()) for text in data]
        self.lda = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=num_topics, workers=4)
        self.indexer = indexer
    
    def get_embeddings(self, text):
        embeddings = self.lda[self.id2word.doc2bow(text.split())]
        return [[embedding[1] for embedding in embeddings]]

    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings)
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)