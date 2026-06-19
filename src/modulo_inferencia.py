from pickle import load
import pandas as pd
from src.pre_processamento import limpar_e_mapear, montar_matriz


def carregar_pkl(nome):
    return load(open(f'./models/{nome}.pkl', 'rb'))


def inferir_rf(dado_bruto):
    modelo = carregar_pkl('classificador_rf')
    ohe = carregar_pkl('OHE_Encoder')

    df = pd.DataFrame([dado_bruto])
    df = limpar_e_mapear(df)
    matriz = montar_matriz(df, ohe)

    pred = modelo.predict(matriz)[0]
    proba = modelo.predict_proba(matriz)[0, 1]

    resultado = 'yes' if pred == 1 else 'no'
    print(f'Predicao: {resultado} (probabilidade de assinar: {proba:.4f})')
    return resultado
