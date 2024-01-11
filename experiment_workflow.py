import logging
import joblib
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'

import numexpr as ne

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm

from utils import *

logger = logging.getLogger(__name__)
logFormatter = '%(asctime)s - %(levelname)s : %(filename)s : %(funcName)s : \
%(lineno)d : %(message)s'
logging.basicConfig(filename='experiment_workflow.log', filemode='w', 
                    format=logFormatter, level=logging.INFO)

logger.info('Loading and preprocessing data...')
try:
    tcu_data = pd.read_csv('./datasets/jurisprudencias_tcu_final_preprocessado.csv')
except:
    tcu_data = pd.read_csv('datasets/jurisprudencias_tcu_final.csv')
    tcu_data.VOTO = tcu_data.VOTO.str.lower()
    tcu_data['PREPROCESSADO'] = textos_preprocessados(tcu_data.VOTO.tolist())
    tcu_data.to_csv('datasets/jurisprudencias_tcu_final_preprocessado.csv')    

try:
    stj_data = pd.read_csv('./datasets/jurisprudencias_stj_final_preprocessado.csv')
except:
    stj_data = pd.read_csv('datasets/jurisprudencias_stj_final.csv')
    stj_data.EMENTA = stj_data.EMENTA.str.lower()
    stj_data['PREPROCESSADO'] = textos_preprocessados(stj_data.EMENTA.tolist())
    stj_data.to_csv('datasets/jurisprudencias_stj_final_preprocessado.csv')    

data = {
    'tcu': {'data': tcu_data, 'texto': 'VOTO', 'num_jurisprudencia': 44},
    'stj': {'data': stj_data, 'texto': 'EMENTA', 'num_jurisprudencia': 1458},
}

embedders_ = embedders()

tfidf = None
tfidf_dictionary = None
results = []

for data_name, items in data.items(): 

    for model_name in tqdm(embedders_.keys()):

        logger.info(model_name.upper())

        if any(substring in model_name for substring in ['tfidf','bm25','weighted','lda']):
            coluna_texto = 'PREPROCESSADO'
        else:
            coluna_texto = items['texto']

        if model_name == 'tfidf':
            tfidf = embedders_[model_name](items['data'][coluna_texto].tolist())
            tfidf_dictionary = dict(zip(tfidf.get_feature_names(),
                        list(tfidf.idf_)))
            model = tfidf
        elif model_name == 'bm25':
            model = embedders_[model_name]([texto.split() 
                                           for texto 
                                           in items['data'][coluna_texto].tolist()])
        elif 'weighted' in model_name:
            model = embedders_[model_name](tfidf, tfidf_dictionary)
        elif 'lda' in model_name:
            model = embedders_[model_name](items['data'][coluna_texto].tolist(), 
                                          items['num_jurisprudencia'])
        else:
            model = embedders_[model_name]()

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
        data.to_csv('results/'+data_name+'/similarities_diff_cse.csv')
# indexer = setup_indexer()
# indexer.load('results/bertlongformer.ann')

logger.info('Finish...')
