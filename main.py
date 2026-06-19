from src.carregar_dados import carregar_dados
from src.pre_processamento import (
    limpar_e_mapear,
    separar_e_dividir,
    fit_ohe,
    montar_matriz
)
from src.balancear import balancear
from src.treinamento_rf import treinar_rf
from src.salvar_modelo import salvar_modelo
from src.avaliacao import avaliador

if __name__ == '__main__':
    dados = carregar_dados()

    # limpeza global + mapeamento de binárias/ordinais (pré-split)
    dados = limpar_e_mapear(dados)

    # separar X/y e split com stratify
    X_train, X_test, y_train, y_test = separar_e_dividir(dados)

    # fit do OHE só no treino
    ohe = fit_ohe(X_train)

    # transform + montagem com np.hstack
    X_train = montar_matriz(X_train, ohe)
    X_test = montar_matriz(X_test, ohe)

    # SMOTE só no treino
    X_train, y_train = balancear(X_train, y_train)

    # RandomizedSearchCV + treino da RF
    modelo = treinar_rf(X_train, y_train)

    # avaliação
    avaliador(modelo, X_test, y_test, nome_modelo='classificador_rf')

    # salvar o encoder
    salvar_modelo(ohe, 'OHE_Encoder')
