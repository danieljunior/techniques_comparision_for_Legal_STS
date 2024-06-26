import logging
import os
import math
import torch
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast, BertForMaskedLM, BertModel
from transformers.modeling_longformer import LongformerSelfAttention
from transformers import TrainingArguments, HfArgumentParser
from utils import BertLong, BertLongSelfAttention, ModelArgs

logger = logging.getLogger(__name__)
logFormatter = '%(asctime)s - %(levelname)s : %(filename)s : %(funcName)s : \
%(lineno)d : %(message)s'
logging.basicConfig(format=logFormatter, level=logging.INFO)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_long_model(bert_path, save_model_to, attention_window, max_pos):
    model = BertModel.from_pretrained(bert_path)
    tokenizer = BertTokenizerFast.from_pretrained(bert_path, model_max_length=max_pos)
    config = model.config

    print(max_pos)
    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size)
    print(new_pos_embed.shape)
    print(model.embeddings.position_embeddings)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)
                      ] = model.embeddings.position_embeddings.weight
        k += step
    print(new_pos_embed.shape)
    model.embeddings.position_ids = torch.from_numpy(
        tf.range(new_pos_embed.shape[0], dtype=tf.int32).numpy()[tf.newaxis, :])
    model.embeddings.position_embeddings = torch.nn.Embedding.from_pretrained(
        new_pos_embed)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn
    print(model.embeddings.position_ids.shape)
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer, new_pos_embed

def run(bert_path: str, output_dir: str):
    logger.info('Starting...')
    parser = HfArgumentParser((TrainingArguments, ModelArgs,))

    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', output_dir,
        '--warmup_steps', '500',
        '--learning_rate', '0.00003',
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        '--max_steps', '3000',
        '--logging_steps', '500',
        '--save_steps', '500',
        '--max_grad_norm', '5.0',
        '--per_gpu_eval_batch_size', '8',
        '--per_gpu_train_batch_size', '2',  # 32GB gpu with fp32
        '--gradient_accumulation_steps', '32',
        '--evaluate_during_training',
        '--do_train',
        '--do_eval',
    ])
    model_path = f'{training_args.output_dir}/bert-base-{model_args.max_pos}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(f'Converting bert-base into bert-base-{model_args.max_pos}')
    model, tokenizer, new_pos_embed = create_long_model(bert_path,
        save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)

    # logger.info(f'Loading the model from {model_path}')
    # tokenizer = BertTokenizerFast.from_pretrained(model_path)
    # model = BertLong.from_pretrained(model_path, output_hidden_states=True)
    # logger.info('Finish...')