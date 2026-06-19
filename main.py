from src.carregar_dados import carregar_dados
from src.preprocessar import (
    limpar_e_mapear,
    separar_e_dividir,
    fit_ohe,
    montar_matriz
)
from src.balancear import balancear
from src.treinar import treinar_rf, salvar_modelo
from src.avaliar import avaliar


if __name__ == '__main__':
    dados = carregar_dados()

    # limpeza e mapeamento de binárias
    dados = limpar_e_mapear(dados)

    # separar X/y e split (stratify)
    X_train, X_test, y_train, y_test = separar_e_dividir(dados)

    # etapa 4 - fit do OHE so no treino
    ohe = fit_ohe(X_train)

    # etapa 4 e 5 - transform + montagem com np.hstack
    X_train = montar_matriz(X_train, ohe)
    X_test = montar_matriz(X_test, ohe)

    # etapa 6 - SMOTE so no treino
    X_train, y_train = balancear(X_train, y_train)

    # etapa 7 - RandomizedSearchCV + treino da RF
    modelo = treinar_rf(X_train, y_train)

    # etapa 8 - avaliacao
    avaliar(modelo, X_test, y_test)

    # salvar o modelo
    salvar_modelo(modelo, ohe)
