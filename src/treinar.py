import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def treinar_rf(X_train, y_train):
    # etapa 7 - grade de hiperparametros + RandomizedSearchCV
    grade = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    busca = RandomizedSearchCV(
        rf, grade, n_iter=10, cv=3,
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    busca.fit(X_train, y_train)

    print(f'Melhores parametros: {busca.best_params_}')
    return busca.best_estimator_


def salvar_modelo(modelo, ohe):
    # etapa 9 - serializacao do modelo e do encoder
    os.makedirs('models', exist_ok=True)

    with open('models/RF_Classifier.pkl', 'wb') as f:
        pickle.dump(modelo, f)

    with open('models/OHE_Encoder.pkl', 'wb') as f:
        pickle.dump(ohe, f)

    print('Modelo e encoder salvos em models/')
