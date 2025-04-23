# Detecção de Fraudes com Modelos de Machine Learning

Este projeto implementa e avalia diferentes algoritmos de Machine Learning para detecção de fraudes em transações financeiras. Os modelos utilizados incluem **XGBoost**, **LightGBM** e **CatBoost**, com e sem validação cruzada. O objetivo é identificar transações fraudulentas em um dataset altamente desbalanceado, utilizando técnicas de balanceamento de dados e otimização de hiperparâmetros.

---

## Estrutura do Projeto

A estrutura do projeto está organizada da seguinte forma:

```
Deteccao-de-Fraudes/
│
├── CatBoost/
│   ├── CatBoost.py                # Modelo CatBoost sem validação cruzada
│   ├── CatBoost_CrossValidation.py # Modelo CatBoost com validação cruzada
│
├── LightGBM/
│   ├── LightGBM.py                # Modelo LightGBM sem validação cruzada
│   ├── LightGBM_CrossValidation.py # Modelo LightGBM com validação cruzada
│
├── XGBoost/
│   ├── XGBoost.py                 # Modelo XGBoost sem validação cruzada
│   ├── XGBoost_CrossValidation.py  # Modelo XGBoost com validação cruzada
│
├── creditcard.csv                 # Dataset de transações financeiras
└── README.md                      # Documentação do projeto
```

---

## Dataset

O dataset utilizado é o **Credit Card Fraud Detection Dataset**, disponível publicamente no [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). Ele contém transações financeiras realizadas por cartões de crédito, com as seguintes características:

- **Classes**: `0` (transação legítima) e `1` (fraude).
- **Desbalanceamento**: Apenas 0,17% das transações são fraudulentas.
- **Features**: 30 variáveis, incluindo `Time`, `Amount` e 28 variáveis anonimizadas (`V1` a `V28`).

---

## Modelos Implementados

### 1. **XGBoost**
- **Descrição**: Algoritmo de gradient boosting eficiente e amplamente utilizado.
- **Técnicas Utilizadas**:
  - Balanceamento de dados com **undersampling** e **SMOTE**.
  - Otimização de hiperparâmetros, como `max_depth`, `eta` e `scale_pos_weight`.
  - Avaliação com métricas como **AUC**, **Average Precision** e **F1-Score**.
- **Arquivos**:
  - `XGBoost.py`: Treinamento sem validação cruzada.
  - `XGBoost_CrossValidation.py`: Treinamento com validação cruzada estratificada (5 folds).

### 2. **LightGBM**
- **Descrição**: Algoritmo de boosting baseado em histogramas, otimizado para grandes datasets.
- **Técnicas Utilizadas**:
  - Balanceamento de dados com **undersampling** e **SMOTE**.
  - Otimização de hiperparâmetros, como `num_leaves`, `learning_rate` e `scale_pos_weight`.
  - Avaliação com métricas como **AUC**, **Average Precision** e **F1-Score**.
- **Arquivos**:
  - `LightGBM.py`: Treinamento sem validação cruzada.
  - `LightGBM_CrossValidation.py`: Treinamento com validação cruzada estratificada (5 folds).

### 3. **CatBoost**
- **Descrição**: Algoritmo de boosting desenvolvido pela Yandex, com suporte nativo para variáveis categóricas.
- **Técnicas Utilizadas**:
  - Balanceamento de dados com **undersampling** e **SMOTE**.
  - Otimização de hiperparâmetros, como `depth`, `learning_rate` e `scale_pos_weight`.
  - Avaliação com métricas como **AUC**, **Average Precision** e **F1-Score**.
- **Arquivos**:
  - `CatBoost.py`: Treinamento sem validação cruzada.
  - `CatBoost_CrossValidation.py`: Treinamento com validação cruzada estratificada (5 folds).

---

## Técnicas Utilizadas

### 1. **Pré-processamento**
- Normalização da variável `Amount` com **StandardScaler**.
- Remoção das variáveis `Time` e `Amount` originais.

### 2. **Balanceamento de Dados**
- **Undersampling**: Redução da classe majoritária para acelerar o treinamento.
- **SMOTE**: Geração de exemplos sintéticos para a classe minoritária.

### 3. **Validação Cruzada**
- Implementada nos scripts `*_CrossValidation.py` para avaliar a robustez dos modelos.

### 4. **Métricas de Avaliação**
- **AUC (Area Under the Curve)**: Mede a capacidade do modelo de separar as classes.
- **Average Precision (AP)**: Avalia o desempenho considerando o desbalanceamento.
- **F1-Score**: Equilíbrio entre precisão e recall.
- **Taxa de Detecção de Fraude** e **Taxa de Falsos Alarmes**.

---

## Resultados

### Métricas Gerais
| Modelo                | AUC   | Average Precision | F1-Score | Taxa de Detecção de Fraude | Taxa de Falsos Alarmes |
|-----------------------|-------|-------------------|----------|----------------------------|------------------------|
| **XGBoost**           | 0.98  | 0.63              | 0.78     | 75.5%                      | 3.0%                   |
| **LightGBM**          | 0.99  | 0.65              | 0.80     | 77.0%                      | 2.5%                   |
| **CatBoost**          | 0.99  | 0.66              | 0.81     | 78.0%                      | 2.3%                   |

### Observações
- Todos os modelos apresentaram excelente desempenho, com **AUC** acima de 0.98.
- O **CatBoost** obteve o melhor equilíbrio entre precisão e recall, com a maior **F1-Score**.
- A validação cruzada garantiu a robustez dos modelos, reduzindo o risco de overfitting.

---

## Visualizações

Os scripts geram os seguintes gráficos para análise:

1. **Matriz de Confusão**: Mostra os verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
2. **Curva ROC**: Avalia a capacidade discriminativa do modelo.
3. **Curva Precision-Recall**: Identifica o melhor threshold para classificação.
4. **Importância das Features**: Mostra as variáveis mais relevantes para o modelo.

---

## Como Executar

1. **Pré-requisitos**:
   - Python 3.8+
   - Bibliotecas: `pandas`, `numpy`, `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `imblearn`, `matplotlib`, `seaborn`.

2. **Instalar Dependências**:
   ```bash
   pip install -r requirements.txt
    ```

3. **Executar os Scripts**:
   - Para treinar um modelo sem validação cruzada:
     ```bash
     python XGBoost/XGBoost.py
     ```
   - Para treinar um modelo com validação cruzada:
     ```bash
     python XGBoost/XGBoost_CrossValidation.py
     ```
   - Para os outros modelos, substitua [XGBoost](http://_vscodecontentref_/0) por [LightGBM](http://_vscodecontentref_/1) ou [CatBoost](http://_vscodecontentref_/2) no caminho do script.

4. **Resultados**:
   - Os gráficos e modelos treinados serão salvos nos diretórios correspondentes ([XGBoost](http://_vscodecontentref_/3), [LightGBM](http://_vscodecontentref_/4), [CatBoost](http://_vscodecontentref_/5)).

   ---


