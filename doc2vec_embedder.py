import gensim

class Doc2VecEmbedder():
    def __init__(self, model_path, indexer):
        self.doc2vec = gensim.models.doc2vec.Doc2Vec.load(model_path)
        self.indexer = indexer
    
    def get_embeddings(self, text):
        embeddings = self.doc2vec.infer_vector(text.split())
        return [embeddings]

    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings)
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)