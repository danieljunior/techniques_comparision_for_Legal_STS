import logging
import pickle

import pandas as pd
from bert_embedder import BertEmbedder
from sentence_transformer_embedder import SentenceTransformerEmbedder
from tfidf_embedder import TfIdfEmbedder
# from rank_bm25 import BM25Plus
from sklearn.metrics.pairwise import cosine_similarity
from numpy import interp
from utils import textos_preprocessados
from tqdm import tqdm

logFormatter = '%(asctime)s - %(levelname)s : %(filename)s : %(funcName)s : \
%(lineno)d : %(message)s'
logging.basicConfig(filename='embedding_heuristic_expert_comparision.log', filemode='w',
                    format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)


def tfidf(data):
    return TfIdfEmbedder(data)


def sentence_transformer():
    return SentenceTransformerEmbedder('models/portuguese_sentence_transformer',
                                       None)


def bert():
    return BertEmbedder('models/bert-base-cased-pt-br', None)


def itd_bert():
    return BertEmbedder('models/itd_bert', None)


heuristic_expert = pd.read_csv('./datasets/heuristic_expert_score.csv', index_col=0)
try:
    with open("datasets/heuristic_expert_processed_texts", "rb") as fp:  # Unpickling
        all_texts = pickle.load(fp)
except:
    all_texts = textos_preprocessados(heuristic_expert['TEXT1'].tolist() + heuristic_expert['TEXT2'].tolist())
    with open("datasets/heuristic_expert_processed_texts", "wb") as fp:  # Pickling
        pickle.dump(all_texts, fp)

embedders = {
    'TFIDF': tfidf,
    'SBERT': sentence_transformer,
    'BERT': bert,
    'ITD_BERT': itd_bert
}

results = {}

for model_name in tqdm(embedders.keys()):

    logger.info(model_name.upper())

    if model_name == 'TFIDF':
        tfidf = embedders[model_name](all_texts)
        tfidf_dictionary = dict(zip(tfidf.get_feature_names(),
                                    list(tfidf.idf_)))
        model = tfidf
    else:
        model = embedders[model_name]()

    for idx, row in heuristic_expert.iterrows():
        embedding_1 = model.get_embeddings(row.TEXT1)[0]
        embedding_2 = model.get_embeddings(row.TEXT2)[0]

        if model_name == 'TFIDF':
            embedding_1 = embedding_1.toarray()[0]
            embedding_2 = embedding_2.toarray()[0]
        else:
            embedding_1 = embedding_1.numpy()
            embedding_2 = embedding_2.numpy()

        embeddings_similarity = cosine_similarity([embedding_1], [embedding_2])[0][0]
        embeddings_similarity = interp(embeddings_similarity, [0, 1], [0, 4])

        if model_name + "_SCORE" in results:
            results[model_name + "_SCORE"].append(embeddings_similarity)
        else:
            results[model_name + "_SCORE"] = [embeddings_similarity]

for model_name, similarity in results.items():
    heuristic_expert[model_name + "_SCORE"] = similarity

logger.info('Saving embeddings similarities...')
heuristic_expert.to_csv('results/heuristic_expert_embeddings_scores.csv', index=False)

logger.info('Saving pearson correlations...')
pearson = heuristic_expert.drop(['TEXT1', 'TEXT2'], axis=1).corr(method='pearson')
pearson.to_csv('results/heuristic_expert_embeddings_pearson.csv', index=False)
spearman = heuristic_expert.drop(['TEXT1', 'TEXT2'], axis=1).corr(method='spearman')
spearman.to_csv('results/heuristic_expert_embeddings_spearman.csv', index=False)

logger.info('Finish!')