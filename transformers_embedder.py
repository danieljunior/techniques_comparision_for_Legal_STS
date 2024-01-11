from annoy import AnnoyIndex
import torch
from transformers import AutoModel, AutoTokenizer

class TransformersEmbedder():
    def __init__(self, model_path, indexer):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.indexer = indexer

    def get_embeddings(self, text):
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        return embeddings

    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings)
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)
