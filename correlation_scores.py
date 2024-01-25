import logging
import pickle

import pandas as pd
# from bert_embedder import BertEmbedder
# from sentence_transformer_embedder import SentenceTransformerEmbedder
# from tfidf_embedder import TfIdfEmbedder
# from rank_bm25 import BM25Plus
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import interp
from utils import textos_preprocessados, embedders
from tqdm import tqdm

logFormatter = '%(asctime)s - %(levelname)s : %(filename)s : %(funcName)s : \
%(lineno)d : %(message)s'
logging.basicConfig(filename='embedding_heuristic_expert_comparision.log', filemode='w',
                    format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data_to_generate_latex(path):
    data = pd.read_csv(path)
    cols = [col for col in data.columns if 'ROUNDED' not in col]
    cols_idx = [idx for idx, col in enumerate(data.columns) if 'ROUNDED' in col]
    tmp = data[cols].rename(columns=lambda x: x.replace('_SCORE', '').upper()).round(2)
    tmp = tmp.drop(cols_idx)
    tmp.insert(loc=0, column='', value=tmp.columns.tolist())
    return tmp

# heuristic_expert = pd.read_csv('./datasets/heuristic_expert_score.csv', index_col=0)
heuristic_expert = pd.read_csv('./results/heuristic_expert_embeddings_scores_v3.csv')
try:
    with open("datasets/heuristic_expert_processed_texts", "rb") as fp:  # Unpickling
        all_texts = pickle.load(fp)
except:
    all_texts = textos_preprocessados(heuristic_expert['TEXT1'].tolist() + heuristic_expert['TEXT2'].tolist())
    with open("datasets/heuristic_expert_processed_texts", "wb") as fp:  # Pickling
        pickle.dump(all_texts, fp)

embedders_ = embedders()

results = {}
tfidf = None
tfidf_dictionary = None

for model_name in tqdm(embedders_.keys()):

    logger.info(model_name.upper())

    if model_name == 'tfidf':
        tfidf = embedders_[model_name](all_texts)
        tfidf_dictionary = dict(zip(tfidf.get_feature_names(),
                                    list(tfidf.idf_)))
        model = tfidf
    elif model_name == 'bm25':
        model = embedders_[model_name]([texto.split()
                                        for texto
                                        in all_texts])
    elif 'weighted' in model_name:
        model = embedders_[model_name](tfidf, tfidf_dictionary)
    elif 'lda' in model_name:
        num_juris_stj = 1458
        model = embedders_[model_name](all_texts, num_juris_stj)
    else:
        model = embedders_[model_name]()

    for idx, row in heuristic_expert.iterrows():
        if model_name != 'bm25':
            embedding_1 = model.get_embeddings(row.TEXT1)[0]
            embedding_2 = model.get_embeddings(row.TEXT2)[0]
        if model_name == 'tfidf':
            embedding_1 = embedding_1.toarray()[0]
            embedding_2 = embedding_2.toarray()[0]
        elif model_name == 'lda':
            embedding_1 = np.array(embedding_1)
            embedding_2 = np.array(embedding_2)
        elif model_name != 'bm25' and not [x for x in ['word2vec', 'fasttext', 'doc2vec']
                                           if x in model_name]:
            embedding_1 = embedding_1.numpy()
            embedding_2 = embedding_2.numpy()

        embeddings_similarity = None
        if model_name != 'bm25':
            embeddings_similarity = cosine_similarity([embedding_1], [embedding_2])[0][0]
            embeddings_similarity = interp(embeddings_similarity, [0, 1], [0, 4])
        else:
            texts = textos_preprocessados([row.TEXT1, row.TEXT2])
            doc_scores = model.get_scores(texts[0])
            doc2_idx = all_texts.index(texts[1])
            embeddings_similarity = doc_scores[doc2_idx] / (doc_scores[doc2_idx] + 10)

        if model_name + "_SCORE" in results:
            results[model_name + "_SCORE"].append(embeddings_similarity)
            results[model_name + "_SCORE_ROUNDED"].append(round(embeddings_similarity, 2))
        else:
            results[model_name + "_SCORE"] = [embeddings_similarity]
            results[model_name + "_SCORE_ROUNDED"] = [round(embeddings_similarity, 2)]

for model_name, similarity in results.items():
    heuristic_expert[model_name] = similarity

logger.info('Saving embeddings similarities...')
heuristic_expert.to_csv('results/heuristic_expert_embeddings_scores_v4.csv', index=False)

logger.info('Saving pearson correlations...')
pearson = heuristic_expert.drop(['TEXT1', 'TEXT2'], axis=1).corr(method='pearson')
pearson.to_csv('results/heuristic_expert_embeddings_pearson_v4.csv', index=False)
spearman = heuristic_expert.drop(['TEXT1', 'TEXT2'], axis=1).corr(method='spearman')
spearman.to_csv('results/heuristic_expert_embeddings_spearman_v4.csv', index=False)

logger.info('Finish!')
