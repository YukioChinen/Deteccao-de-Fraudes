# CatBoost com validação cruzada para detecção de fraudes

import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# 1. Carregar os dados
print("Carregando dados...")
start_time = time.time()
data = pd.read_csv('creditcard.csv')
print(f"Dados carregados em {time.time() - start_time:.2f} segundos")

# 2. Análise exploratória simples
print(data['Class'].value_counts(normalize=True) * 100)

# 3. Pré-processamento
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop(['Class', 'Time', 'Amount'], axis=1)
X['Amount_scaled'] = data['Amount_scaled']
y = data['Class']

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Validação cruzada com CatBoost
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
average_precision_scores = []

params = {
    'iterations': 5000,
    'learning_rate': 0.01,
    'depth': 8,
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'early_stopping_rounds': 100,
    'verbose': 100,
    'task_type': 'CPU',
    'bootstrap_type': 'Bayesian'
}

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
    print(f"\n--- Fold {fold} ---")

    X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    n_fraud = y_fold_train.value_counts()[1]
    fraud_idx = y_fold_train[y_fold_train == 1].index
    non_fraud_idx = y_fold_train[y_fold_train == 0].index
    sampled_non_fraud_idx = np.random.choice(non_fraud_idx, n_fraud * 10, replace=False)
    under_idx = np.concatenate([fraud_idx, sampled_non_fraud_idx])
    X_fold_under = X_fold_train.loc[under_idx]
    y_fold_under = y_fold_train.loc[under_idx]

    sm = SMOTE(random_state=42)
    X_fold_res, y_fold_res = sm.fit_resample(X_fold_under, y_fold_under)

    train_pool = cb.Pool(X_fold_res, y_fold_res)
    val_pool = cb.Pool(X_fold_val, y_fold_val)

    params['scale_pos_weight'] = sum(y_fold_res == 0) / sum(y_fold_res == 1)

    model = cb.CatBoost(params)
    model.fit(train_pool, eval_set=val_pool, verbose_eval=100, use_best_model=True)

    y_val_proba = model.predict(X_fold_val, prediction_type='Probability')[:, 1]
    auc = roc_auc_score(y_fold_val, y_val_proba)
    ap = average_precision_score(y_fold_val, y_val_proba)

    auc_scores.append(auc)
    average_precision_scores.append(ap)

    print(f"AUC Fold {fold}: {auc:.4f}")
    print(f"Average Precision Fold {fold}: {ap:.4f}")

print(f"\nAUC médio: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Average Precision médio: {np.mean(average_precision_scores):.4f} ± {np.std(average_precision_scores):.4f}")

# 6. Treinamento final com todo o treino balanceado
print("\nTreinando modelo final com todo o treino balanceado...")
n_fraud = y_train.value_counts()[1]
fraud_indices = y_train[y_train == 1].index
non_fraud_indices = y_train[y_train == 0].index
random_non_fraud_indices = np.random.choice(non_fraud_indices, n_fraud * 10, replace=False)
under_sample_indices = np.concatenate([fraud_indices, random_non_fraud_indices])
X_train_under = X_train.loc[under_sample_indices]
y_train_under = y_train.loc[under_sample_indices]

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_under, y_train_under)

final_model = cb.CatBoost(params)
train_dataset = cb.Pool(X_train_res, y_train_res)
eval_dataset = cb.Pool(X_test, y_test)
final_model.fit(train_dataset, eval_set=eval_dataset, verbose_eval=100, use_best_model=True)

# 7. Avaliação
print("\nAvaliando modelo final...")
y_pred_proba = final_model.predict(X_test, prediction_type='Probability')[:, 1]
average_precision = average_precision_score(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC: {auc:.4f}')
print(f'Average Precision: {average_precision:.4f}')

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f'Threshold ótimo baseado em F1: {optimal_threshold:.4f}')

y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)

print("\nResultados com threshold otimizado:")
print(cm_optimal)
print(classification_report(y_test, y_pred_optimal))

tn, fp, fn, tp = cm_optimal.ravel()
fraud_detection_rate = tp / (tp + fn)
false_alarm_rate = fp / (fp + tn)
print(f"Taxa de detecção de fraude: {fraud_detection_rate:.4f}")
print(f"Taxa de falsos alarmes: {false_alarm_rate:.4f}")

# Visualizações
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusão (threshold={optimal_threshold:.3f})')
plt.xlabel('Predito')
plt.ylabel('Real')

plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(recall, precision, label=f'AP = {average_precision:.3f}')
plt.axvline(recall[optimal_idx], color='r', linestyle='--', label=f'Threshold: {optimal_threshold:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()

plt.subplot(2, 2, 4)
feature_importance = final_model.get_feature_importance()
sorted_idx = np.argsort(feature_importance)[-10:]
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [X_test.columns[i] for i in sorted_idx])
plt.title('Importância das Features')

# Criar diretório de saída
output_dir = 'CatBoost'
os.makedirs(output_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'catboost_crossvalidation_results.png'))
plt.show()

# Análise de erros
false_positives = X_test[(y_pred_optimal == 1) & (y_test == 0)]
false_negatives = X_test[(y_pred_optimal == 0) & (y_test == 1)]

print(f"\nNúmero de falsos positivos: {len(false_positives)}")
print(f"Número de falsos negativos: {len(false_negatives)}")

# Salvar o modelo
final_model.save_model(os.path.join(output_dir, 'catboost_crossvalidation_model.cbm'))
print("catboost_crossvalidation_model.cbm'")
