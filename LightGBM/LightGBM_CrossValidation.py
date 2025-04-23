# LightGBM para detecção de fraudes com Cross Validation e otimizações

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
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
print(data['Class'].value_counts(normalize=True) * 100)  # Percentual de fraudes

# 3. Pré-processamento
# Normalizar os valores do Amount
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# 4. Separar features e target
X = data.drop(['Class', 'Time', 'Amount'], axis=1)  # Removendo Time e Amount original
X['Amount_scaled'] = data['Amount_scaled']  # Adicionando Amount normalizado
y = data['Class']

# 5. Definir Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

y_true_all = []
y_pred_proba_all = []
auc_scores = []
ap_scores = []
f1_scores = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"\nTreinando fold {fold}...")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Balanceamento - undersampling + SMOTE
    n_fraud = y_train.value_counts()[1]
    fraud_indices = y_train[y_train == 1].index
    non_fraud_indices = y_train[y_train == 0].index
    random_non_fraud_indices = np.random.choice(non_fraud_indices, n_fraud * 10, replace=False)
    under_sample_indices = np.concatenate([fraud_indices, random_non_fraud_indices])
    X_train_under = X_train.loc[under_sample_indices]
    y_train_under = y_train.loc[under_sample_indices]

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_under, y_train_under)

    # Treinamento LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 40,
        'min_data_in_leaf': 60,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'scale_pos_weight': sum(y_train_res == 0) / sum(y_train_res == 1)
    }

    lgb_train = lgb.Dataset(X_train_res, y_train_res)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    model = lgb.train(params,
                      lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      valid_names=['train', 'valid'],
                      num_boost_round=5000,
                      callbacks=[])

    # Avaliação do modelo
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    average_precision = average_precision_score(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_fold = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1 = np.nanmax(f1_fold)
    optimal_idx = np.argmax(f1_fold)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

    y_true_all.extend(y_test)
    y_pred_proba_all.extend(y_pred_proba)
    auc_scores.append(auc)
    ap_scores.append(average_precision)
    f1_scores.append(best_f1)

# Conversão para array numpy
y_true_all = np.array(y_true_all)
y_pred_proba_all = np.array(y_pred_proba_all)

# Cálculo de métricas finais
overall_auc = roc_auc_score(y_true_all, y_pred_proba_all)
overall_ap = average_precision_score(y_true_all, y_pred_proba_all)
overall_precision, overall_recall, overall_thresholds = precision_recall_curve(y_true_all, y_pred_proba_all)
overall_f1_scores = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-10)
overall_optimal_idx = np.argmax(overall_f1_scores)
overall_optimal_threshold = overall_thresholds[overall_optimal_idx]
y_pred_optimal = (y_pred_proba_all >= overall_optimal_threshold).astype(int)
overall_cm = confusion_matrix(y_true_all, y_pred_optimal)

# Gráficos finais
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusão (threshold={overall_optimal_threshold:.3f})')
plt.xlabel('Predito')
plt.ylabel('Real')

plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_true_all, y_pred_proba_all)
plt.plot(fpr, tpr, label=f'AUC = {overall_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(overall_recall, overall_precision, label=f'AP = {overall_ap:.3f}')
plt.axvline(overall_recall[overall_optimal_idx], color='r', linestyle='--', label=f'Threshold = {overall_optimal_threshold:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()

# Criar diretório de saída
output_dir = 'LightGBM'
os.makedirs(output_dir, exist_ok=True)

# Salvar figura
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'crossval_lightgbm_results.png'))
plt.show()

# Resultados finais
print("\n=== Resultados Médios ===")
print(f"Média AUC por fold: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Média AP por fold: {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")
print(f"Média F1-score por fold: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"AUC total: {overall_auc:.4f}")
print(f"Average Precision total: {overall_ap:.4f}")
print(f"Threshold ótimo total: {overall_optimal_threshold:.4f}")

print("\nClassification Report Geral:")
print(classification_report(y_true_all, y_pred_optimal))

# Salvar o modelo
model.save_model(os.path.join(output_dir, 'lightgbm_crossvalidation_model.json'))
print("Modelo salvo em 'lightgbm_crossvalidation_model.json'")