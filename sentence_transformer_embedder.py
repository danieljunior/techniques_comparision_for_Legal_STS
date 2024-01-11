from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import torch

class SentenceTransformerEmbedder():
    def __init__(self, model_path, indexer):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_path, device=device)
        # self.model.max_seq_length = 512
        self.indexer = indexer

    def get_embeddings(self, text):
        embeddings = self.model.encode([text], convert_to_tensor=True)
        return embeddings

    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings)
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)
