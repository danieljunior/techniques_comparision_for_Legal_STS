import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    data = pd.read_csv('datasets/jurisprudencias_stj.csv', index_col=0)
    v = data[['jurisprudencia_index']]

    # only data from jurisprudencies with more than 1 
    custom_data = data[v.replace(v.apply(pd.Series.value_counts)).gt(2).all(1)]
    X_train, X_test, y_train, y_test = train_test_split(custom_data.ementa, 
                                                        custom_data.jurisprudencia_index, 
                                                        test_size=0.2, 
                                                        stratify=custom_data.jurisprudencia_index, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test