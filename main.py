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
from utils import prepare_data

logger = logging.getLogger(__name__)
logFormatter = '%(asctime)s - %(levelname)s : %(filename)s : %(funcName)s : \
%(lineno)d : %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)

def setup_indexer(vectors_size=3072):
    return AnnoyIndex(vectors_size, 'angular')

logger.info('Loading data...')
X_train, X_test, y_train, y_test = prepare_data()

logger.info('Loading models...')

# tfidf = TfIdfEmbedder(X_train[:10])
# tfidf.set_indexer(setup_indexer(vectors_size=len(tfidf.get_feature_names())))
# dictionary = dict(zip(tfidf.get_feature_names(),
#                       list(tfidf.idf_)))


# word2vec = Word2VecEmbedder('models/word2vec/skip_s300.txt', 
#                             indexer = setup_indexer(vectors_size=300))

# weighted_word2vec = Word2VecEmbedder('models/word2vec/skip_s300.txt', 
#                             tfidf_model=tfidf, 
#                             tfidf_dictionary=dictionary,
#                             indexer = setup_indexer(vectors_size=300))

# fasttext = Word2VecEmbedder('models/fasttext/skip_s300.txt', 
#                             indexer = setup_indexer(vectors_size=300))

# weighted_fasttext = Word2VecEmbedder('models/fasttext/skip_s300.txt', 
#                             tfidf_model=tfidf, 
#                             tfidf_dictionary=dictionary,
#                             indexer = setup_indexer(vectors_size=300))

sentence_transformer = SentenceTransformerEmbedder('distiluse-base-multilingual-cased-v2', 
                                                   setup_indexer(vectors_size=512))
# options_path = 'models/elmo/options.json'
# weights_path = 'models/elmo/elmo_pt_weights_dgx1.hdf5'
# elmo = ElmoEmbedder(options_path, weights_path, setup_indexer(vectors_size=1024))

# bert = BertEmbedder('models/bert-base-cased-pt-br', setup_indexer())
# longformer = LongformerEmbedder('models/bert_longformer', setup_indexer())
# itd_longformer = LongformerEmbedder('models/itd_bert_longformer', setup_indexer())

# TODO falta o finetunning_bert, bm25

embedders = {
    # 'tfidf': tfidf,
    # 'word2vec': word2vec,
    # 'weighted_word2vec': weighted_word2vec,    
    # 'fasttext': fasttext,
    # 'weighted_fasttext': weighted_fasttext,
    'sentence_transformer': sentence_transformer
    # 'elmo': elmo, 
    # 'bert':bert,
    # 'longformer':longformer,
    # 'itd_longformer': itd_longformer
}

for k,v in tqdm(embedders.items()):
    model = v
    for i, ementa in tqdm(enumerate(X_train[:10])):
        # import pdb; pdb.set_trace()
        embeddings = model.get_embeddings(ementa)[0]
        model.add_to_indexer(i, embeddings)

    model.save_indexer('results/'+k+'.ann')

# index = 0
# indexer.add_item(index, embeddings)

# indexer.build(10) # 10 trees
# indexer.save('results/bertlongformer.ann')

# indexer = setup_indexer()
# indexer.load('results/bertlongformer.ann')

# nns = indexer.get_nns_by_item(index, 5)
# a = indexer.get_item_vector(index)

# for n in nns[1:]:
#         b = indexer.get_item_vector(n)
#         similaridade = cosine_similarity([a], [b])[0][0]
#         print(similaridade)
logger.info('Finish...')
