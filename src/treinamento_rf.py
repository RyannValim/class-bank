import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.salvar_modelo import salvar_modelo

def treinar_rf(X_train, y_train):
    # definindo os hiperparâmetros
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_depth = [None] + [int(x) for x in np.linspace(start=5, stop=80, num=16)]
    criterion = ['gini', 'entropy']
    min_samples_split = [2, 4, 6, 8, 10, 15, 20]
    min_samples_leaf = [1, 2, 4, 6, 8]
    max_features = ['sqrt', 'log2', None]
    class_weight = ['balanced', 'balanced_subsample', None]
    bootstrap = [True, False]
    n_jobs = -1
    random_state = 42
    verbose = 2
    cv = 5
    n_iter = 15

    # distribuição de parâmetros em grid
    rf_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'class_weight': class_weight,
        'bootstrap': bootstrap,
    }

    # instanciando o estimador
    classificador_rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

    # treinando a base inteira com cross validation
    hiperparametros_rf = RandomizedSearchCV(
        estimator=classificador_rf,
        param_distributions=rf_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state
    )

    hiperparametros_rf.fit(X_train, y_train)

    # salva o melhor modelo
    salvar_modelo(hiperparametros_rf.best_estimator_, 'classificador_rf')

    # devolve a melhor combinação de valores
    return hiperparametros_rf.best_estimator_
