import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(test_size=0.2):
    data = pd.read_csv('datasets/jurisprudencias_stj.csv', index_col=0)
    v = data[['jurisprudencia_index']]

    # only data from jurisprudencies with more than 1 
    custom_data = data[v.replace(v.apply(pd.Series.value_counts)).gt(2).all(1)]
    custom_data.to_csv('datasets/custom_data.csv')
    train, test = train_test_split(custom_data, 
                            test_size=test_size, 
                            stratify=custom_data.jurisprudencia_index,
                            shuffle=True,
                            random_state=42)
    return train, test