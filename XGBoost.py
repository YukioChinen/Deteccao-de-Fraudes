# XGBoost para detecção de fraudes com otimizações

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
print("Carregando dados...")
start_time = time.time()
data = pd.read_csv('creditcard.csv')
print(f"Dados carregados em {time.time() - start_time:.2f} segundos")

# 2. Análise exploratória simples
print(data['Class'].value_counts(normalize=True) * 100)  # Percentual de fraudes

# 3. Pré-processamento
# Normalizar os valores do Amount
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# 3. Separar features e target
X = data.drop(['Class', 'Time', 'Amount'], axis=1)  # Removendo Time e Amount original
X['Amount_scaled'] = data['Amount_scaled']  # Adicionando Amount normalizado
y = data['Class']

# 4. Dividir em treino e teste - estratificado para preservar a proporção de fraudes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Balanceamento dos dados - usando undersampling + SMOTE
# Reduzir o número de exemplos da classe majoritária para agilizar o treinamento
n_fraud = y_train.value_counts()[1]
fraud_indices = y_train[y_train == 1].index
non_fraud_indices = y_train[y_train == 0].index
random_non_fraud_indices = np.random.choice(non_fraud_indices, n_fraud * 10, replace=False)
under_sample_indices = np.concatenate([fraud_indices, random_non_fraud_indices])
X_train_under = X_train.loc[under_sample_indices]
y_train_under = y_train.loc[under_sample_indices]

# Aplicar SMOTE no conjunto reduzido
print("Aplicando SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_under, y_train_under)

print(f'Dataset original completo: {y_train.value_counts()}')
print(f'Dataset após undersampling: {y_train_under.value_counts()}')
print(f'Dataset após SMOTE: {y_train_res.value_counts()}')

# 6. Treinamento com XGBoost
print("Treinando modelo XGBoost...")
# Criando DMatrix
xg_train = xgb.DMatrix(X_train_res, label=y_train_res)
xg_test = xgb.DMatrix(X_test, label=y_test)

# Parâmetros otimizados
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.01,              # Equivalente a learning_rate
    'max_depth': 8,
    'min_child_weight': 60,   # Similar a min_data_in_leaf
    'subsample': 0.8,         # Similar a bagging_fraction
    'colsample_bytree': 0.8,  # Similar a feature_fraction
    'scale_pos_weight': sum(y_train_res == 0) / sum(y_train_res == 1),  # Balanceamento de classes
    'tree_method': 'hist',    # Para acelerar o treinamento
    'verbosity': 0
}

watchlist = [(xg_train, 'train'), (xg_test, 'valid')]

# Treinamento
start_time = time.time()
model = xgb.train(
    params, 
    xg_train, 
    num_boost_round=5000, 
    evals=watchlist, 
    early_stopping_rounds=100,
    verbose_eval=100
)
print(f"Modelo treinado em {time.time() - start_time:.2f} segundos")

# 7. Avaliação do modelo com threshold otimizado
y_pred_proba = model.predict(xg_test, iteration_range=(0, model.best_iteration + 1))

# Calcular precisão média (AP) e AUC
average_precision = average_precision_score(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC: {auc:.4f}')
print(f'Average Precision: {average_precision:.4f}')

# Encontrar o melhor threshold baseado na curva de precisão-recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
# Calculando F1 para cada threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f'Threshold ótimo baseado em F1: {optimal_threshold:.4f}')

# Resultados com threshold otimizado
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)

# Imprimir resultados com threshold otimizado
print("\nResultados com threshold otimizado:")
print(confusion_matrix(y_test, y_pred_optimal))
print(classification_report(y_test, y_pred_optimal))

# Métricas específicas para detecção de fraude
tn, fp, fn, tp = cm_optimal.ravel()
# Taxa de detecção de fraude
fraud_detection_rate = tp / (tp + fn)
# Taxa de falsos alarmes
false_alarm_rate = fp / (fp + tn)
print(f"Taxa de detecção de fraude: {fraud_detection_rate:.4f}")
print(f"Taxa de falsos alarmes: {false_alarm_rate:.4f}")

# Visualizações
plt.figure(figsize=(12, 10))

# 1. Matriz de confusão
plt.subplot(2, 2, 1)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusão (threshold={optimal_threshold:.3f})')
plt.xlabel('Predito')
plt.ylabel('Real')

# 2. Curva ROC
plt.subplot(2, 2, 2)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()

# 3. Curva Precision-Recall
plt.subplot(2, 2, 3)
plt.plot(recall, precision, label=f'AP = {average_precision:.3f}')
plt.axvline(recall[optimal_idx], color='r', linestyle='--', 
            label=f'Threshold: {optimal_threshold:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()

# 4. Importância das features
plt.subplot(2, 2, 4)
xgb.plot_importance(model, max_num_features=10, importance_type='gain', ax=plt.gca())
plt.title('Importância das Features')

plt.tight_layout()
plt.savefig('xgboost_results.png')
plt.show()

# Análise de erros - examinar falsos positivos e falsos negativos
false_positives = X_test[(y_pred_optimal == 1) & (y_test == 0)]
false_negatives = X_test[(y_pred_optimal == 0) & (y_test == 1)]

print(f"\nNúmero de falsos positivos: {len(false_positives)}")
print(f"Número de falsos negativos: {len(false_negatives)}")

# Salvar o modelo
model.save_model('xgboost_fraud_model.json')
print("Modelo salvo em 'xgboost_fraud_model.json'")