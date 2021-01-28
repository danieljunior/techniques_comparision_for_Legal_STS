#https://github.com/allenai/allennlp/issues/2245
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch

class ElmoEmbedder():
    def __init__(self, options_path, weights_path, indexer):
        self.elmo = Elmo(options_path, weights_path, 1, dropout=0)
        self.indexer = indexer
    
    def get_embeddings(self, text):
        sentences = [text.split(' ')]
        character_ids = batch_to_ids(sentences)
        elmo_embeddings = self.elmo(character_ids)
        tokens_embeddings = elmo_embeddings["elmo_representations"][0][0]
        embeddings = torch.mean(tokens_embeddings, dim=0)
        return [embeddings] 
    
    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings)
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)