from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def run(model_path: str, model_name: str, train_file: str, eval_file: str, output_path: str, block_size: str = 512):
    cuda_available = torch.cuda.is_available()
    model_args = LanguageModelingArgs()
    model_args.config = {
        "sliding_window": True,
        'block_size': block_size,
        "save_steps": -1,
        "save_model_every_epoch": False
    }
    model = LanguageModelingModel(model_name, model_path, args=model_args, use_cuda=cuda_available)
    model.train_model(train_file, output_dir=output_path, eval_file=eval_file)


def prepare_data(output_path: str, data: pd.Series):
    train_file = output_path+'/train.txt'
    test_file = output_path+'/test.txt'

    train, test = train_test_split(data, shuffle=True)
    with open(train_file, 'w') as f:
        for item in train:
            f.write("%s\n" % item)

    with open(test_file, 'w') as f:
        for item in test:
            f.write("%s\n" % item)
    
    return train_file, test_file