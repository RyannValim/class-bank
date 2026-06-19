import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

def avaliador(modelo, X_test, y_test, nome_modelo='modelo'):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    # acurácia
    acuracia = accuracy_score(y_test, y_pred)

    # sensibilidade e especificidade via matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensibilidade = tp / (tp + fn)
    especificidade = tn / (tn + fp)

    roc_auc = roc_auc_score(y_test, y_proba)

    print(f'\n--- Avaliação: {nome_modelo} ---')
    print(f'Acurácia:        {acuracia:.4f}')
    print(f'Sensibilidade:   {sensibilidade:.4f}')
    print(f'Especificidade:  {especificidade:.4f}')
    print(f'ROC-AUC:         {roc_auc:.4f}')

    # matriz de confusão visual
    os.makedirs('./plots', exist_ok=True)
    ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test)
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.savefig(f'./plots/{nome_modelo}_confusion_matrix.png')
    plt.close()

    return {
        'acuracia': acuracia,
        'sensibilidade': sensibilidade,
        'especificidade': especificidade,
        'roc_auc': roc_auc
    }
