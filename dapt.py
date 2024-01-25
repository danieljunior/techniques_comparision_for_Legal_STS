import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import ray
import numpy as np
import random

from longformer_embedder import LongformerEmbedder
from bert_embedder import BertEmbedder

random.seed(42)


def count_tokens(data, tokenizer, text_columns, output_path):
    results = []
    for index, doc in tqdm(data.iterrows()):
        tokens_A = len(tokenizer.tokenize(doc[text_columns[0]]))
        tokens_B = len(tokenizer.tokenize(doc[text_columns[1]]))
        results.append([tokens_A, tokens_B])
    ta = [t[0] for t in results]
    tb = [t[1] for t in results]
    data['n_tokens_a'] = ta
    data['n_tokens_b'] = tb
    data.to_csv(output_path, index=False)


def main_count_tokens():
    print("Loading data...")
    # stj_data = pd.read_csv('./datasets/stj_sts.csv')
    tcu_data = pd.read_csv('./datasets/tcu_sts.csv')

    print("Loading tokenizer...")
    tokenizer_path = './models/bert-base-cased-pt-br'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)

    print("Counting tokens...")
    count_tokens(tcu_data, tokenizer, ['sentence_A', 'sentence_B'], './datasets/tcu_sts_with_tokens.csv')
    print("Finish")


def filter_n_tokens(data, bottom_bound, upper_bound, ):
    return data[(data.n_tokens_a >= bottom_bound) & (data.n_tokens_a <= upper_bound) &
                (data.n_tokens_b >= bottom_bound) & (data.n_tokens_b <= upper_bound)]


def bert():
    return BertEmbedder('models/bert-base-cased-pt-br', None)


def itd_bert():
    return BertEmbedder('models/itd_bert', None)


def longformer():
    return LongformerEmbedder('models/bert_longformer', None)


def itd_longformer():
    return LongformerEmbedder('models/itd_bert_longformer', None)


@ray.remote
def calculate_similarities(sample, embedder_name):
    # get_worker_id
    pid = ray.get_runtime_context().get_worker_id()
    embedder = embedders[embedder_name]()
    results = []
    for index, doc in tqdm(sample.iterrows(), desc=str(pid) + " Documents pairs"):
        embeddings_A = embedder.get_embeddings(doc.sentence_A)[0]
        embeddings_B = embedder.get_embeddings(doc.sentence_B)[0]
        similarity = cosine_similarity([embeddings_A.numpy()], [embeddings_B.numpy()])[0][0]
        results.append(similarity)
    return results


def calculate_similarities_by_length_range():
    # 1) Textos menores que 512 tokens (mais próximo da metade)
    # 2) Textos até 512 tokens (mais próximo disso possível)
    # 3) Textos maiores que 512 tokens (maior que isso possível)

    data = pd.read_csv('./datasets/stj_sts_with_tokens.csv')
    minor = filter_n_tokens(data, 206, 306)
    medium = filter_n_tokens(data, 412, 512)
    large = filter_n_tokens(data, 612, 10000)

    min_length = min([len(minor), len(medium), len(large)])
    # min_length = 5

    samples = {'minor': minor.sample(min_length, random_state=42),
               'medium': medium.sample(min_length, random_state=42),
               'large': large.sample(min_length, random_state=42)}

    embedders = {'bert': bert,
                 'itd_bert': itd_bert,
                 'longformer': longformer,
                 'itd_longformer': itd_longformer}

    # ray.init(num_cpus=4, ignore_reinit_error=True)
    ray.init(ignore_reinit_error=True)

    for sample_name, sample in tqdm(samples.items(), desc="Samples"):
        results = {}
        processes = []

        for embedder_name in tqdm(embedders.keys(), desc="Embedders"):
            processes.append(calculate_similarities.remote(sample, embedder_name))

        for idx, process in enumerate(processes):
            result = ray.get(process)
            embedder_name = list(embedders.keys())[idx]
            results[embedder_name] = result
        embedders_names = list(results.keys())
        for name in embedders_names:
            sample[name.upper() + "_SIMILARITY"] = results[name]
        sample.to_csv('./datasets/stj_sts_' + sample_name + ".csv", index=False)


def calculate_correlation_between_methods_in_differents_length_ranges():
    for variation in ['minor', 'medium', 'large']:
        print(variation.upper())
        data = pd.read_csv('./datasets/stj_sts_' + variation + '.csv')
        data['score'] = data['score'].apply(lambda x: np.interp(x, [0, 5], [0, 1]))
        data = (data.rename(columns=lambda x: x.replace('_SIMILARITY', ''))
                .rename(columns=lambda x: 'HEURISTIC' if x == 'score' else x))
        data = data.drop(['sentence_A', 'sentence_B', 'range', 'SPLIT', 'n_tokens_a', 'n_tokens_b'], axis=1)
        pearson = data.corr(method='pearson').round(2)
        print("\nPEARSON")
        print(pearson)
        pearson.to_csv('./results/' + variation + 'stj_dapt_long_pearson.csv', index=False)
        spearman = data.corr(method='spearman').round(2)
        print("\nSPEARMAN")
        print(spearman)
        spearman.to_csv('./results/' + variation + 'stj_dapt_long_spearman.csv', index=False)
        print("#############################################################################")


def count_words_length():
    print("Loading data...")
    for dataname in ['stj', 'tcu']:
        print(dataname.upper())
        data = pd.read_csv('./datasets/jurisprudencias_' + dataname + '_final.csv')
        column = 'EMENTA' if dataname == 'stj' else 'VOTO'
        length = np.array([len(x.split(' ')) for x in data[column]])
        print("MEAN {}".format(length.mean().round(0)))
        print("STD {}".format(length.std().round(0)))


calculate_correlation_between_methods_in_differents_length_ranges()
