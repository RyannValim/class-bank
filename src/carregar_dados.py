import pandas as pd

def carregar_dados():
    return pd.read_csv('./data/raw/bank.csv', sep=';')