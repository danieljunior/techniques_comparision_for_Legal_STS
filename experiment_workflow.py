import logging
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from longformer_embedder import LongformerEmbedder
from bert_embedder import BertEmbedder
from word2vec_embedder import Word2VecEmbedder
from tfidf_embedder import TfIdfEmbedder
from elmo_embedder import ElmoEmbedder
from sentence_transformer_embedder import SentenceTransformerEmbedder
from tqdm import tqdm
from utils import prepare_data, bm25_tokenizer, biggests_index
from rank_bm25 import BM25Plus

logger = logging.getLogger(__name__)
logFormatter = '%(asctime)s - %(levelname)s : %(filename)s : %(funcName)s : \
%(lineno)d : %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)

def setup_indexer(vectors_size=3072):
    return AnnoyIndex(vectors_size, 'angular')

def tfidf(data):    
    tfidf = TfIdfEmbedder(data)
    tfidf.set_indexer(setup_indexer(vectors_size=len(tfidf.get_feature_names())))
    dictionary = dict(zip(tfidf.get_feature_names(),
                        list(tfidf.idf_)))
    return tfidf, dictionary

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
    return BM25Plus(bm25_tokenizer(data))

logger.info('Loading data...')
train, test = prepare_data(test_size=0.5)

embedders = {
    'tfidf': tfidf,
    'bm25': bm25,
    'word2vec': word2vec,
    'weighted_word2vec': weighted_word2vec,    
    'fasttext': fasttext,
    'weighted_fasttext': weighted_fasttext,
    'sentence_transformer': sentence_transformer,
    'elmo': elmo, 
    'bert': bert,
    'itd_bert': itd_bert,
    'longformer':longformer,
    'itd_longformer': itd_longformer
}

tfidf = None
tfidf_dictionary = None
results = []

for model_name in tqdm(embedders.keys()):

    logger.info(model_name.upper())
    
    if model_name == 'tfidf':
        tfidf, tfidf_dictionary = embedders[model_name](train.ementa.tolist())
        model = tfidf
    elif model_name == 'bm25':
        model = embedders[model_name](train.ementa.tolist())
    elif 'weighted' in model_name:
        model = embedders[model_name](tfidf, tfidf_dictionary)
        
    if model_name != 'bm25':
        logger.info('Getting embeddings and add to indexer...')
        for index, doc in tqdm(train.iterrows()):
            embeddings = model.get_embeddings(doc.ementa)[0]
            model.add_to_indexer(index, embeddings)

        model.save_indexer('results/'+model_name+'.ann')
    
    logger.info('Getting neighbors...')
    
    for source_index, doc in tqdm(train.iterrows()):
        
        if model_name != 'bm25':
            nns = model.indexer.get_nns_by_item(source_index, 6)
            source_vector = model.indexer.get_item_vector(source_index)
            
            for similar_index in nns[1:]:
                similar_vector = model.indexer.get_item_vector(similar_index)
                similarity = cosine_similarity([source_vector], [similar_vector])[0][0]
                results.append([source_index, similar_index, similarity, model_name])    

        else:
            doc_scores = model.get_scores(doc.ementa.split(' '))
            nns = biggests_index(doc_scores, 5)
            for similar_index in nns:
                #src: https://stats.stackexchange.com/questions/171589/normalised-score-for-bm25
                normalized_similarity = doc_scores[similar_index]/sum(doc_scores[nns])
                results.append([source_index, similar_index, normalized_similarity, model_name])
            
logger.info('Saving results...')
data = pd.DataFrame(results, 
                    columns=['SOURCE_INDEX','SIMILAR_INDEX','COSINE_SIMILARITY','MODEL_NAME'])
data.to_csv('results/similarities.csv')
# indexer = setup_indexer()
# indexer.load('results/bertlongformer.ann')

logger.info('Finish...')
