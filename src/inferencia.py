import pickle

import pandas as pd

from src.preprocessar import limpar_e_mapear, montar_matriz


def carregar_artefatos():
    # carrega modelo e encoder salvos (sem retreinar)
    with open('models/RF_Classifier.pkl', 'rb') as f:
        modelo = pickle.load(f)

    with open('models/OHE_Encoder.pkl', 'rb') as f:
        ohe = pickle.load(f)

    return modelo, ohe


def inferir(dado_bruto):
    # dado_bruto = dict com as colunas originais (sem o y)
    modelo, ohe = carregar_artefatos()

    # vira um DataFrame de uma linha
    df = pd.DataFrame([dado_bruto])

    # reproduz etapa 1 e 2 (limpeza global + mapeamento)
    df = limpar_e_mapear(df)

    # reproduz etapa 4 e 5 (OHE transform + np.hstack)
    matriz = montar_matriz(df, ohe)

    pred = modelo.predict(matriz)[0]
    proba = modelo.predict_proba(matriz)[0, 1]

    resultado = 'yes' if pred == 1 else 'no'
    print(f'Predicao: {resultado} (probabilidade de assinar: {proba:.4f})')
    return resultado


if __name__ == '__main__':
    # exemplo de um novo dado bruto (mesmas colunas do CSV original, sem o y)
    novo_dado = {
        'age': 42,
        'job': 'management',
        'marital': 'married',
        'education': 'tertiary',
        'default': 'no',
        'balance': 1500,
        'housing': 'yes',
        'loan': 'no',
        'contact': 'cellular',
        'day': 15,
        'month': 'may',
        'duration': 200,
        'campaign': 2,
        'pdays': -1,
        'previous': 0,
        'poutcome': 'unknown'
    }

    inferir(novo_dado)
