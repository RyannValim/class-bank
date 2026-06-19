import os
from pickle import dump

def salvar_modelo(modelo, nome_modelo):
    os.makedirs('./models', exist_ok=True)
    dump(modelo, open(f'./models/{nome_modelo}.pkl', 'wb'))
