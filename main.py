import joblib
from tfidf_embedder import TfIdfEmbedder

model = joblib.load('results/results/tcu_tfidf.joblib')
print('TFIDF Vector size:')
print(len(model.get_feature_names()))