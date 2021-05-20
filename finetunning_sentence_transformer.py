"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.
Usage:
python training_nli.py
OR
python training_nli.py pretrained_transformer_model_name
"""
import sys
import os
import logging
from datetime import datetime
import math
import xml.dom.minidom

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
from tqdm import tqdm
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def read_sick_dataset(dataset_path: str = 'data/SICK_BR.txt'):
    # Convert the dataset to a DataLoader ready for training
    logging.info("Read SICK_BR dataset")

    data = pd.read_csv(dataset_path, delimiter='\t')

    train_samples = []
    dev_samples = []
    test_samples = []

    for i, row in tqdm(data.iterrows()):
        score = float(row.relatedness_score) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row.sentence_A, row.sentence_B], label=score)

        if row.SemEval_set == 'TRIAL':
            dev_samples.append(inp_example)
        elif row.SemEval_set == 'TEST':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

    return train_samples, dev_samples, test_samples

def process_assin_xml(doc):
    result = []
    for pair in tqdm(doc.getElementsByTagName('pair')):
        score = pair.getAttribute('similarity')
        score = float(score) / 5.0  # Normalize score to range 0 ... 1
        sentence_A = pair.getElementsByTagName('h')[0].firstChild.nodeValue
        sentence_B = pair.getElementsByTagName('t')[0].firstChild.nodeValue
        example = InputExample(texts=[sentence_A, sentence_B], label=score)
        result.append(example)
    return result

def read_assin_dataset(dataset_path: str = 'datasets/ASSIN', prefix:str = 'assin-pt-br'):
    logging.info("Read ASSIN dataset")
    
    train = xml.dom.minidom.parse(dataset_path+prefix+'-train.xml')
    dev = xml.dom.minidom.parse(dataset_path+prefix+'-dev.xml')
    test = xml.dom.minidom.parse(dataset_path+prefix+'-test.xml')
    train_samples = process_assin_xml(train)
    dev_samples = process_assin_xml(dev)
    test_samples = process_assin_xml(test)

    return train_samples, dev_samples, test_samples

def run(train_samples, dev_samples, test_samples, model_path: str, train_batch_size: int = 16,
        num_epochs: int = 4, model_save_path:str = "output/custom-sentence-transformer"):
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_path)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataset = SentencesDataset(train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))
    logging.info("Trainning...")
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)

    logging.info("Test evaluating...")
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    test_evaluator(model, output_path=model_save_path)

train1, dev1, test1 = read_sick_dataset('./datasets/SICK_BR.txt')
train2, dev2, test2 = read_assin_dataset('./datasets/ASSIN/pt-br/','assin-ptbr')
train3, dev3, test3 = read_assin_dataset('./datasets/ASSIN 2/','assin2')
train_samples = train1+train2+train3
dev_samples = dev1+dev2+dev3
test_samples = test1+test2+test3
BERT_PATH = 'models/bert-base-cased-pt-br'
logging.info('Train samples: %i'%len(train_samples))
logging.info('Dev samples: %i'%len(dev_samples))
logging.info('Test samples: %i'%len(test_samples))
run(train_samples, dev_samples, test_samples, BERT_PATH, 16, 4, "models/portuguese_sentence_transformer")