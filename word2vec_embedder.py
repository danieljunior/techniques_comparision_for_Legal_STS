# source: https://github.com/v1shwa/document-similarity/blob/master/DocSim.py
import numpy as np
from gensim import models
import json
import pickle

class Word2VecEmbedder():
    
    def __init__(self, model_path, tfidf_model=None, tfidf_dictionary=None, indexer=None):
        self.w2v_model = models.KeyedVectors.load_word2vec_format(model_path)
        self.tfidf_model = tfidf_model
        self.tfidf_dictionary = tfidf_dictionary
        self.indexer = indexer
    
    def build_model(self, sentences):
        self.w2v_model = models.Word2Vec(sentences=sentences, min_count=50, size=300, workers=4)

    def get_embeddings(self, doc: str) -> np.ndarray:
        """
        Identify the vector values for each word in the given document
        :param doc:
        :return:
        """
        if self.tfidf_model and self.tfidf_dictionary:
            return self.tfidf_weighted_vectorize(doc)

        return self.centroid_vectorize(doc)

    def centroid_vectorize(self, doc: str) -> np.ndarray:
        """
        Identify the vector values for each word in the given document
        :param doc:
        :return:
        """
        word_vecs = []
        for word in doc:
            try:
                vec = self.w2v_model.wv[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        return [vector]

    def tfidf_weighted_vectorize(self, doc: str) -> np.ndarray:
        names = self.tfidf_model.get_feature_names()
        word_vecs = []
        weight_sum = 0
        w2vwords = list(self.w2v_model.wv.vocab)
        for word in doc:
            try:
                if word in names and word in w2vwords:
                    vec = self.w2v_model.wv[word]
                    tfidf = self.tfidf_dictionary[word] * (doc.count(word)/len(doc))
                    word_vecs.append(vec * tfidf)
                    weight_sum+= tfidf
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass
        if weight_sum != 0:
            word_vecs /= weight_sum

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        return [vector]

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, source_doc, target_docs=None, threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if not target_docs:
            return []

        if isinstance(target_docs, str):
            target_docs = [target_docs]

        source_vec = self.get_embeddings(source_doc)
        results = []
        for doc in target_docs:
            target_vec = self.get_embeddings(doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                results.append({"score": sim_score, "doc": doc})
            # Sort results by score in desc order
            results.sort(key=lambda k: k["score"], reverse=True)

        return results

    def vector_size(self):
        return self.w2v_model.vector_size

    def save(self, path='model.bin'):
        self.w2v_model.save(path)
        if self.tfidf_model and self.tfidf_dictionary:
            with open(path+'_dict.json', 'w') as f:
                json_string = json.dumps(self.tfidf_dictionary)
                f.write(json_string)
            with open(path+'_tfidf.bin', 'wb') as f:
                pickle.dump(self.tfidf_model, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path='model.bin'):
        self.w2v_model = models.Word2Vec.load(path)
        if self.tfidf_model and self.tfidf_dictionary:
            with open(path+'_dict.json', 'r') as f:
                self.tfidf_dictionary = json.loads(f.read())
            with open(path+'_tfidf.bin', 'rb') as f:
                self.tfidf_model = pickle.load(f)
    
    def add_to_indexer(self, index, embeddings):
        self.indexer.add_item(index, embeddings)
    
    def save_indexer(self, path):
        self.indexer.build(10) # 10 trees
        self.indexer.save(path)
