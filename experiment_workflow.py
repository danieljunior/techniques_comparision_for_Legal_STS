import logging
import joblib
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'

import numexpr as ne
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from longformer_embedder import LongformerEmbedder
from bert_embedder import BertEmbedder
from word2vec_embedder import Word2VecEmbedder
from tfidf_embedder import TfIdfEmbedder
from elmo_embedder import ElmoEmbedder
from sentence_transformer_embedder import SentenceTransformerEmbedder
from doc2vec_embedder import Doc2VecEmbedder
from lda_embedder import LDAEmbedder
from tqdm import tqdm
from utils import tokenize, biggests_index, textos_preprocessados
from rank_bm25 import BM25Plus
import ray

logger = logging.getLogger(__name__)
logFormatter = '%(asctime)s - %(levelname)s : %(filename)s : %(funcName)s : \
%(lineno)d : %(message)s'
logging.basicConfig(filename='experiment_workflow.log', filemode='w', 
                    format=logFormatter, level=logging.INFO)

def setup_indexer(vectors_size=3072):
    return AnnoyIndex(vectors_size, 'angular')

def tfidf(data):    
    tfidf = TfIdfEmbedder(data)
    tfidf.set_indexer(setup_indexer(vectors_size=len(tfidf.get_feature_names())))
    return tfidf

def word2vec():
    return Word2VecEmbedder('models/word2vec/skip_s300.txt', 
                            indexer = setup_indexer(vectors_size=300))

def weighted_word2vec(tfidf, dictionary):
    return Word2VecEmbedder('models/word2vec/skip_s300.txt', 
                            tfidf_model=tfidf, 
                            tfidf_dictionary=dictionary,
                            indexer = setup_indexer(vectors_size=300))

def fasttext():
    return Word2VecEmbedder('models/fasttext/skip_s300.txt', 
                            indexer = setup_indexer(vectors_size=300))
    
def weighted_fasttext(tfidf, dictionary):
    return Word2VecEmbedder('models/fasttext/skip_s300.txt', 
                            tfidf_model=tfidf, 
                            tfidf_dictionary=dictionary,
                            indexer = setup_indexer(vectors_size=300))
def lda(data, num_topics=44):
    return LDAEmbedder(data, 
                       indexer=setup_indexer(vectors_size=num_topics), 
                       num_topics=num_topics)
    
def doc2vec():
    return Doc2VecEmbedder('models/itd_doc2vec_model',
                           indexer = setup_indexer(vectors_size=100))

def sentence_transformer():
    return SentenceTransformerEmbedder('distiluse-base-multilingual-cased-v2', 
                                        setup_indexer(vectors_size=512))

def elmo():
    options_path = 'models/elmo/options.json'
    weights_path = 'models/elmo/elmo_pt_weights_dgx1.hdf5'
    return ElmoEmbedder(options_path, weights_path, setup_indexer(vectors_size=1024))

def bert():
    return BertEmbedder('models/bert-base-cased-pt-br', setup_indexer())

def itd_bert():
    return BertEmbedder('models/itd_bert', setup_indexer())

def longformer(): 
    return LongformerEmbedder('models/bert_longformer', setup_indexer())

def itd_longformer():
    return LongformerEmbedder('models/itd_bert_longformer', setup_indexer())

def bm25(data):
    return BM25Plus(data)

logger.info('Loading and preprocessing data...')
try:
    tcu_data = pd.read_csv('datasets/jurisprudencias_tcu_final_preprocessado.csv')
except:
    tcu_data = pd.read_csv('datasets/jurisprudencias_tcu_final.csv')
    tcu_data.VOTO = tcu_data.VOTO.str.lower()
    tcu_data['PREPROCESSADO'] = textos_preprocessados(tcu_data.VOTO.tolist())
    tcu_data.to_csv('datasets/jurisprudencias_tcu_final_preprocessado.csv')    

try:
    stj_data = pd.read_csv('datasets/jurisprudencias_stj_final_preprocessado.csv')
except:
    stj_data = pd.read_csv('datasets/jurisprudencias_stj_final.csv')
    stj_data.EMENTA = stj_data.EMENTA.str.lower()
    stj_data['PREPROCESSADO'] = textos_preprocessados(stj_data.EMENTA.tolist())
    stj_data.to_csv('datasets/jurisprudencias_stj_final_preprocessado.csv')    

data = {
    'tcu': {'data': tcu_data.sample, 'texto': 'VOTO', 'num_jurisprudencia': 44},
    'stj': {'data': stj_data.sample, 'texto': 'EMENTA', 'num_jurisprudencia': 1458},
}

embedders = {
    'tfidf': tfidf,
    'bm25': bm25,
    'lda': lda,
    'word2vec': word2vec,
    'weighted_word2vec': weighted_word2vec,    
    'fasttext': fasttext,
    'weighted_fasttext': weighted_fasttext,
    'doc2vec': doc2vec,
    'sentence_transformer': sentence_transformer,
    'bert': bert,
    'itd_bert': itd_bert,
    'longformer':longformer,
    'itd_longformer': itd_longformer,
    'elmo': elmo, 
}

tfidf = None
tfidf_dictionary = None
results = []

for data_name, items in data.items(): 

    for model_name in tqdm(embedders.keys()):

        logger.info(model_name.upper())

        if any(substring in model_name for substring in ['tfidf','bm25','weighted','lda']):
            coluna_texto = 'PREPROCESSADO'
        else:
            coluna_texto = items['texto']

        if model_name == 'tfidf':
            tfidf = embedders[model_name](items['data'][coluna_texto].tolist())
            tfidf_dictionary = dict(zip(tfidf.get_feature_names(),
                        list(tfidf.idf_)))
            model = tfidf
        elif model_name == 'bm25':
            model = embedders[model_name]([texto.split() 
                                           for texto 
                                           in items['data'][coluna_texto].tolist()])
        elif 'weighted' in model_name:
            model = embedders[model_name](tfidf, tfidf_dictionary)
        elif 'lda' in model_name:
            model = embedders[model_name](items['data'][coluna_texto].tolist(), 
                                          items['num_jurisprudencia'])
        else:
            model = embedders[model_name]()

        if model_name != 'bm25':
            logger.info('Getting embeddings and add to indexer...')
            for index, doc in tqdm(items['data'].iterrows()):
                    embeddings = model.get_embeddings(doc[coluna_texto])[0]
                    model.add_to_indexer(index, embeddings)
            model.save_indexer('results/'+data_name+'/'+model_name+'.ann')

        if model_name in ['bm25','tfidf']:
            joblib.dump(model, 'results/'+data_name+'/'+model_name+'.joblib')

            # some time later...

            # load the model from disk
            # loaded_model = joblib.load(filename)

        logger.info('Getting neighbors...')

        for source_index, doc in tqdm(items['data'].iterrows()):

            if model_name != 'bm25':
                nns = model.indexer.get_nns_by_item(source_index, 6)
                source_vector = model.indexer.get_item_vector(source_index)

                for similar_index in nns[1:]:
                    similar_vector = model.indexer.get_item_vector(similar_index)
                    similarity = cosine_similarity([source_vector], [similar_vector])[0][0]
                    results.append([source_index, similar_index, similarity, model_name])    

            else:
                doc_scores = model.get_scores(doc[coluna_texto].split(' '))
                nns = biggests_index(doc_scores, 5)
                for similar_index in nns:
                    #src: https://stats.stackexchange.com/questions/171589/normalised-score-for-bm25
                    normalized_similarity = doc_scores[similar_index]/sum(doc_scores[nns])
                    results.append([source_index, similar_index, normalized_similarity, model_name])
            
        logger.info('Saving results...')
        data = pd.DataFrame(results, 
                            columns=['SOURCE_INDEX','SIMILAR_INDEX','COSINE_SIMILARITY','MODEL_NAME'])
        data.to_csv('results/'+data_name+'/similarities.csv')
# indexer = setup_indexer()
# indexer.load('results/bertlongformer.ann')

logger.info('Finish...')
