import joblib
from tfidf_embedder import TfIdfEmbedder
from bert_embedder import BertEmbedder
from elmo_embedder import ElmoEmbedder
from sentence_transformer_embedder import SentenceTransformerEmbedder
import pandas as pd
from tqdm import tqdm
from utils import tokenize
import seaborn as sns
import matplotlib.pyplot as plt

# model = joblib.load('results/results/tcu_tfidf.joblib')
# print('TFIDF Vector size:')
# print(len(model.get_feature_names()))

def tfidf():    
    return joblib.load('results/tcu_tfidf.joblib')

def sentence_transformer():
    return SentenceTransformerEmbedder('distiluse-base-multilingual-cased-v2', 
                                        None)

def elmo():
    options_path = 'models/elmo/options.json'
    weights_path = 'models/elmo/elmo_pt_weights_dgx1.hdf5'
    return ElmoEmbedder(options_path, weights_path, None)

def bert():
    return BertEmbedder('models/bert-base-cased-pt-br', None)

embedders = {
    'tfidf': tokenize,
    'sentence_transformer': sentence_transformer,
    'bert': bert,
    'others': None
}

tcu_data = pd.read_csv('datasets/jurisprudencias_tcu_final.csv')

resp = {}
for model_name in tqdm(embedders.keys()):
    if model_name in ['bert','sentence_transformer']:
        model = embedders[model_name]()
    model_tokens = []
    for i, row in tqdm(tcu_data.iterrows()):
        if model_name=='bert':
            tokens = len(model.tokenizer.tokenize(row.VOTO))
        elif model_name=='sentence_transformer':
            tokens = len(model.model.tokenizer.tokenize(row.VOTO))
        elif model_name == 'tfidf':
            tokens = len(tokenize(row.VOTO))
        else:
            tokens = len(row.VOTO.split(' '))            
        model_tokens.append(tokens)
    resp[model_name] = model_tokens

data_values = []
for model, values in resp.items():
    for v in values:
        data_values.append([model, v])

data = pd.DataFrame(data_values, columns=['MODEL TOKENIZER','TOKENS NUMBERS'])
ax = sns.boxplot(x="MODEL TOKENIZER", y="TOKENS NUMBERS", data=data)
plt.savefig('results/tokens_boxplot.png',bbox_inches='tight')