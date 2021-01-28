from transformers import BertTokenizerFast
from models.bertlongformer import BertLong, get_features, get_concat_four_last_layers
from annoy import AnnoyIndex

class LongformerEmbedder():
    def __init__(self, model_path, indexer):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertLong.from_pretrained(model_path, output_hidden_states=True)
        self.indexer = indexer

    def get_embeddings(self, text):
        last_hidden_state, pooler_output, hidden_states = get_features(text, 
                                                                       self.model, 
                                                                       self.tokenizer)
        embeddings = get_concat_four_last_layers(hidden_states)
        return embeddings

    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings)
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)
