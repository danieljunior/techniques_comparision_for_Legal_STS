import logging
import torch
import nltk
from nltk import tokenize
nltk.download('punkt')
from transformers import BertModel
from transformers.modeling_longformer import LongformerSelfAttention
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class BertLong(BertModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)


@dataclass
class ModelArgs:
    attention_window: int = field(
        default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})


def convert_examples_to_features(example, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    tokens = ['[CLS]']
    for i, w in enumerate(tokenize.word_tokenize(example, language='portuguese')):
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        tokens.extend(sub_words)

    # truncate
    if len(tokens) > seq_length - 1:
        logger.info('Example is too long, length is {}, truncated to {}!'.format(
            len(tokens), seq_length))
        tokens = tokens[0:(seq_length - 1)]
    tokens.append('[SEP]')

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask], dtype=torch.long)
    
    return input_ids, segment_ids, input_mask


def get_features(input_text, model, tokenizer, dim=768, max_length=4096):
    text_split = get_split(input_text, max_length=max_length//2)
    resp = []
    for text in text_split:
        input_ids, segment_ids, input_mask = convert_examples_to_features(
            example=text, seq_length=max_length, tokenizer=tokenizer)
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=segment_ids,
                            attention_mask=input_mask)
            # last_hidden_state, pooler_output, hidden_states
        resp.append(outputs)
    return resp 


def get_last_layer(hidden_states):
    return hidden_states[-1]

def get_summed_four_last_layers(hidden_states):
    return torch.stack(hidden_states[-4:]).sum(0)

def get_concat_four_last_layers(hidden_states):
    pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
    return pooled_output[:, 0, :]


def ceiling_division(n, d):
    return -(n // -d)

def get_split(text, max_length=200, overlap=50):
  l_total = []
  l_parcial = []
  text_len = len(text.split())
  aux_value = (max_length - overlap)
  splits = ceiling_division(text_len,aux_value)
  if splits > 0:
    n = splits
  else: 
    n = 1
  for w in range(n):
    if w == 0:
      l_parcial = text.split()[:max_length]
      l_total.append(" ".join(l_parcial))
    else:
      l_parcial = text.split()[w*aux_value:w*aux_value + max_length]
      l_total.append(" ".join(l_parcial))
  return l_total