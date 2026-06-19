from src.modulo_inferencia import inferir_rf

# Exemplo de cliente para inferência
# Preencha os campos abaixo com os dados reais do cliente
dado = {
    'age': 35,
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
    'campaign': 2,
    'pdays': -1,
    'previous': 0,
    'poutcome': 'unknown',
}

if __name__ == '__main__':
    inferir_rf(dado)
