from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)


def avaliar(modelo, X_test, y_test):
    # etapa 8 - avaliacao com os dados puros do teste
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    acuracia = accuracy_score(y_test, y_pred)

    # sensibilidade (recall da classe positiva)
    sensibilidade = recall_score(y_test, y_pred)

    # especificidade = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    especificidade = tn / (tn + fp)

    roc_auc = roc_auc_score(y_test, y_proba)

    print(f'Acuracia:        {acuracia:.4f}')
    print(f'Sensibilidade:   {sensibilidade:.4f}')
    print(f'Especificidade:  {especificidade:.4f}')
    print(f'ROC-AUC:         {roc_auc:.4f}')
