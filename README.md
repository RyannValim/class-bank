# bank-classifier

Classificador do dataset Bank Marketing utilizando **Random Forest** com busca de hiperparâmetros via `RandomizedSearchCV`, balanceamento de classes com **SMOTE** e codificação de variáveis categóricas com **OneHotEncoder**.

## Estrutura do projeto

```
class-bank/
├── data/
│   ├── raw/            # dataset original (bank.csv)
│   └── processed/      # dados pós-processamento (gerado automaticamente)
├── models/             # modelos serializados em .pkl (gerado automaticamente)
├── plots/              # gráficos de avaliação (gerado automaticamente)
├── src/
│   ├── carregar_dados.py       # leitura do CSV
│   ├── pre_processamento.py    # limpeza, mapeamento e OHE
│   ├── balancear.py            # SMOTE no conjunto de treino
│   ├── treinamento_rf.py       # RandomizedSearchCV + RandomForest
│   ├── avaliacao.py            # métricas e matriz de confusão
│   ├── salvar_modelo.py        # serialização com pickle
│   └── modulo_inferencia.py    # pipeline de inferência
├── main.py             # pipeline de treinamento completo
├── inferencia.py       # script de inferência para novos dados
└── requirements.txt
```

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Uso

### Treinamento

Coloque o arquivo `bank.csv` em `data/raw/` e execute:

```bash
python main.py
```

Ao final, os arquivos `models/classificador_rf.pkl` e `models/OHE_Encoder.pkl` serão gerados, junto com a matriz de confusão em `plots/`.

### Inferência

Edite o dicionário `dado` em `inferencia.py` com os atributos do cliente e execute:

```bash
python inferencia.py
```

A saída indica se o cliente tende a assinar o depósito (`yes`/`no`) e a probabilidade associada.

## Pipeline

| Etapa | Descrição |
|---|---|
| Limpeza | Remove `duration` (data leakage) e converte `pdays` em flag binária |
| Mapeamento | Binárias (`yes`/`no` → 0/1), ordinais (`education`, `month`) |
| Split | 70/30 com `stratify` |
| OHE | Fit apenas no treino; `handle_unknown='ignore'` |
| SMOTE | Balanceamento apenas no treino |
| RandomForest | `RandomizedSearchCV` com 30 iterações, CV=5, scoring=`roc_auc` |
| Avaliação | Acurácia, sensibilidade, especificidade e ROC-AUC |

## Métricas avaliadas

- **Acurácia** — proporção de predições corretas
- **Sensibilidade** — capacidade de identificar clientes que assinarão (`TP / (TP + FN)`)
- **Especificidade** — capacidade de rejeitar clientes que não assinarão (`TN / (TN + FP)`)
- **ROC-AUC** — área sob a curva ROC
