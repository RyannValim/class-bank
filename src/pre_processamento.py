import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# mapeamentos diretos
MAPA_BINARIO = {'yes': 1, 'no': 0}
MAPA_EDUCATION = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
MAPA_MONTH = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

# grupos de colunas
COLS_BINARIAS = ['default', 'housing', 'loan']
COLS_NUMERICAS = ['age', 'balance', 'campaign', 'previous', 'day']
COLS_OHE = ['job', 'marital', 'contact', 'poutcome']


def limpar_e_mapear(dados):
    dados = dados.copy()

    # dropar duration (causa data leakage) — pode já estar ausente na inferência
    dados = dados.drop(columns=['duration'], errors='ignore')

    # pdays -1 (nunca contatado) vira flag binaria de contato previo
    dados['contactado_antes'] = (dados['pdays'] != -1).astype(int)
    dados = dados.drop(columns=['pdays'])

    # mapear target (so existe no treino, na inferencia nao tem)
    if 'y' in dados.columns:
        dados['y'] = dados['y'].map(MAPA_BINARIO)

    # mapear binarias
    for col in COLS_BINARIAS:
        dados[col] = dados[col].map(MAPA_BINARIO)

    # mapear ordinais
    dados['education'] = dados['education'].map(MAPA_EDUCATION)
    dados['month'] = dados['month'].map(MAPA_MONTH)

    os.makedirs('./data/processed', exist_ok=True)
    dados.to_csv('./data/processed/bank_processed.csv', index=False)

    return dados


def separar_e_dividir(dados):
    X = dados.drop(columns=['y'])
    y = dados['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def fit_ohe(X_train):
    # fit do OHE so no treino (aprende as categorias, inclui unknown)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train[COLS_OHE])
    return ohe


def montar_matriz(X, ohe):
    ohe_array = ohe.transform(X[COLS_OHE])

    binarias = X[COLS_BINARIAS].values
    ordinais = X[['education', 'month']].values
    flag = X[['contactado_antes']].values
    numericas = X[COLS_NUMERICAS].values

    # empilha: binarias + ordinais + flag + numericas + OHE
    matriz = np.hstack([binarias, ordinais, flag, numericas, ohe_array])
    return matriz
