import logging
import joblib
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'

import pandas as pd
from tqdm import tqdm
from sklearn.svm import LinearSVR
import numpy as np

from utils import *

logger = logging.getLogger(__name__)
logFormatter = '%(asctime)s - %(levelname)s : %(filename)s : %(funcName)s : \
%(lineno)d : %(message)s'
logging.basicConfig(filename='supervised_workflow.log', filemode='w', 
                    format=logFormatter, level=logging.INFO)


logger.info('Loading STS data...')
tcu_data = pd.read_csv('datasets/tcu_sts.csv')
stj_data = pd.read_csv('datasets/stj_sts.csv')

datasets = {
    'tcu': {'data': tcu_data, 'num_jurisprudencia': 44},
    'stj': {'data': stj_data, 'num_jurisprudencia': 1458},
}

num_jurisprudencia = {}

embedders_ = embedders()
del embedders_['bm25']

tfidf = None
tfidf_dictionary = None


for data_name, items in datasets.items():
    logger.info(f'Running experiment with {data_name} dataset...')

    dataset = items['data'] 
    train_data = dataset[dataset.SPLIT == 'TRAIN']
    test_data = dataset[dataset.SPLIT != 'TRAIN']
    
    for embedder_name in tqdm(list(embedders_.keys())[:1]):
        logger.info('Setup embedder...')
        sentences = list(set(list(train_data['sentence_A'] 
                                            + train_data['sentence_B'])))
        if embedder_name == 'tfidf':
            
            tfidf = embedders_[embedder_name](sentences)
            tfidf_dictionary = dict(zip(tfidf.get_feature_names(),
                        list(tfidf.idf_)))
            embedder = tfidf
        elif embedder_name == 'bm25':
            embedder = embedders_[embedder_name]([texto.split() 
                                           for texto 
                                           in sentences])
        elif 'weighted' in embedder_name:
            embedder = embedders_[embedder_name](tfidf, tfidf_dictionary)
        elif 'lda' in embedder_name:
            embedder = embedders_[embedder_name](sentences, 
                                          items['num_jurisprudencia'])
        else:
            embedder = embedders_[embedder_name]()

        X, y = [], []
        logger.info('Loading train data...')
        for index, doc in tqdm(train_data.iterrows()):
            x1 = embedder.get_embeddings(doc['sentence_A'])[0]
            x2 = embedder.get_embeddings(doc['sentence_B'])[0]
            
            X.append(np.concatenate([x1, x2], axis=0))
            y.append(doc['score'])
        
        logger.info('Trainning regressor...')
        regressor = LinearSVR(random_state=0, tol=1e-05)
        regressor.fit(X, y)
        print(f'Predição: {regressor.predict(X[:10])}')
        print(f'Real: {y[:10]}')
