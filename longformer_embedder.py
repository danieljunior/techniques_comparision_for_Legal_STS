from transformers import BertTokenizerFast
from bertlongformer import BertLong, get_features, get_concat_four_last_layers
from annoy import AnnoyIndex
import torch

class LongformerEmbedder():
    def __init__(self, model_path, indexer):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertLong.from_pretrained(model_path, output_hidden_states=True)
        self.indexer = indexer

    def get_embeddings(self, text):
        outputs = get_features(text, 
                                   self.model, 
                                   self.tokenizer)
        embeddings = []
        for last_hidden_state, pooler_output, hidden_states in outputs:
            embeddings.append(get_concat_four_last_layers(hidden_states))
        
        return torch.mean(torch.stack(embeddings), dim=0)

    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings)
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)
