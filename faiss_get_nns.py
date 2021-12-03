from annoy import AnnoyIndex
import faiss                   # make faiss available
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import pandas as pd

datasets = ['tcu', 'stj']
embedders = {
    'tfidf':{'tcu': 34081, 'stj': 25696},
    'lda':{'tcu': 44, 'stj': 1458},
    'word2vec': 300,
    'weighted_word2vec': 300,    
    'fasttext': 300,
    'weighted_fasttext': 300,
    'doc2vec': 100,
    'sentence_transformer': 768,
    'bert': 3072,
    'itd_bert': 3072,
    'longformer':3072,
    'itd_longformer': 3072,
    'elmo': 1024, 
}

def convert_annoy_to_faiss():
    for data_name in datasets:
        for model_name, model_data in embedders.items():
            print(model_name)
            indexer_path = './results/'+data_name+'/'+model_name+'.ann'
            vector_size = None
            if model_name in ['tfidf','lda']:
                vector_size = model_data[data_name]
            else:
                vector_size = model_data
                
            indexer = AnnoyIndex(vector_size, 'angular')
            indexer.load(indexer_path)
            
            nlist = 100 #number of cluster centers 
            quantizer = faiss.IndexFlat(vector_size)  # the other index
            if model_name == 'lda':
                metric = faiss.METRIC_JensenShannon
            else:
                metric = faiss.METRIC_INNER_PRODUCT

            faiss_indexer = faiss.IndexIVFFlat(quantizer, vector_size, nlist, metric)
            faiss_indexer.set_direct_map_type(faiss.DirectMap.Array)
            embeddings = []
            ids = []
            for i in tqdm(range(indexer.get_n_items())):
                vector = indexer.get_item_vector(i)
    #             norm_vector = faiss.normalize_L2(np.array([vector]).astype('float32'))
                norm_vector = np.array(vector).astype('float32')
                embeddings.append(norm_vector)
                ids.append(i)
            
            faiss_indexer.train(np.array(embeddings).astype('float32'))
            faiss_indexer.add(np.array(embeddings).astype('float32'))
            faiss.write_index(faiss_indexer, './results/faiss/'+data_name+'/'+model_name+'.faiss')
            
def get_nns():
    datasets_len = {'tcu':371,'stj':7403}
    k = 6

    for data_name in datasets:
        results = []
        for model_name, model_data in embedders.items():
            print(model_name)
            indexer = faiss.read_index('./results/faiss/'+data_name+'/'+model_name+'.faiss')
            indexer.nprobe = 100
            for source_index in tqdm(range(datasets_len[data_name])):
                source_vector = indexer.reconstruct(source_index)
                D, I  = indexer.search(np.array([source_vector]).astype('float32'), k)
                for similar_index in I[0][1:]:
                    similar_vector = indexer.reconstruct(int(similar_index))
                    if model_name != 'lda':
                        similarity = cosine_similarity([source_vector], [similar_vector])[0][0]
                    else:
                        similarity = distance.jensenshannon(source_vector, similar_vector)
                        
                    results.append([source_index, similar_index, similarity, model_name])
        data = pd.DataFrame(results, 
                                columns=['SOURCE_INDEX','SIMILAR_INDEX','SIMILARITY','MODEL_NAME'])
        data.to_csv('results/faiss/'+data_name+'/similarities.csv')

get_nns()