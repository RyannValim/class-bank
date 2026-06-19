from imblearn.over_sampling import SMOTE


def balancear(X_train, y_train):
    # etapa 6 - SMOTE estritamente no treino
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    return X_bal, y_bal
