"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.
Usage:
python training_nli.py
OR
python training_nli.py pretrained_transformer_model_name
"""

import logging
import xml.dom.minidom

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
from tqdm import tqdm
import pandas as pd
import numpy as np

torch.manual_seed(42)
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def read_sick_dataset(dataset_path: str = 'data/SICK_BR.txt', split="train"):
    # Convert the dataset to a DataLoader ready for training
    data = pd.read_csv(dataset_path, delimiter='\t')
    train_sentences = data[data.SemEval_set == 'TRAIN']
    train_sentences = np.unique(train_sentences.sentence_A + train_sentences.sentence_B)
    return [InputExample(texts=[s, s]) for s in train_sentences]

def read_assin_dataset(dataset_path: str = 'datasets/ASSIN', prefix: str = 'assin-pt-br'):
    logging.info("Read ASSIN dataset")

    train = xml.dom.minidom.parse(dataset_path + prefix + '-train.xml')
    sentences = []
    for pair in tqdm(train.getElementsByTagName('pair')):
        # score = pair.getAttribute('similarity')
        # score = float(score) / 5.0  # Normalize score to range 0 ... 1
        sentences.append(pair.getElementsByTagName('h')[0].firstChild.nodeValue)
        sentences.append(pair.getElementsByTagName('t')[0].firstChild.nodeValue)
    sentences = np.unique(sentences)
    return [InputExample(texts=[s, s]) for s in sentences]

if __name__ == "__main__":
    base_model_path = "models/bert-base-cased-pt-br"
    model_output_path = "models/portuguese_sim_cse"
    num_epochs = 4
    train_batch_size = 16

    logging.info("Reading SICK_BR dataset...")
    sick_examples = read_sick_dataset('./datasets/SICK_BR.txt', split="train")
    logging.info("Reading ASSIN 1 dataset...")
    assin1_examples = read_assin_dataset('./datasets/ASSIN/pt-br/','assin-ptbr')
    logging.info("Reading ASSIN 2 dataset...")
    assin2_examples = read_assin_dataset('./datasets/ASSIN 2/','assin2')
    train_examples = sick_examples + assin1_examples + assin2_examples
    dataloader = DataLoader(train_examples, batch_size=128, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_embedding_model = models.Transformer(base_model_path, max_seq_length=32)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    logging.info("Trainning...")
    model.fit(train_objectives=[(dataloader, train_loss)],
              epochs=1)

    logging.info("Saving model...")
    model.save(model_output_path)
    logging.info("Finish!")

