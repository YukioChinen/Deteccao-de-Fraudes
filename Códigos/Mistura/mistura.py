# Combinação de XGBoost e LightGBM para detecção de fraudes

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
print("Carregando dados...")
start_time_load = time.time()
data = pd.read_csv('creditcard.csv')
print(f"Dados carregados em {time.time() - start_time_load:.2f} segundos")

# 2. Análise exploratória simples
print("Distribuição das classes:")
print(data['Class'].value_counts(normalize=True) * 100)

# 3. Pré-processamento
print("Pré-processando dados...")
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
X = data.drop(['Class', 'Time', 'Amount'], axis=1)
X['Amount_scaled'] = data['Amount_scaled']
y = data['Class']

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Tamanho do conjunto de treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")

# 5. Balanceamento dos dados (Undersampling + SMOTE)
print("Balanceando dados de treino...")
start_time_balance = time.time()
n_fraud = y_train.value_counts()[1]
fraud_indices = y_train[y_train == 1].index
non_fraud_indices = y_train[y_train == 0].index
# Reduzir majoritária para 10x o número de fraudes antes do SMOTE
random_non_fraud_indices = np.random.choice(non_fraud_indices, min(len(non_fraud_indices), n_fraud * 10), replace=False)
under_sample_indices = np.concatenate([fraud_indices, random_non_fraud_indices])
X_train_under = X_train.loc[under_sample_indices]
y_train_under = y_train.loc[under_sample_indices]

# Aplicar SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_under, y_train_under)
print(f"Balanceamento concluído em {time.time() - start_time_balance:.2f} segundos")
print(f'Dataset original treino: {y_train.value_counts()}')
print(f'Dataset após undersampling: {y_train_under.value_counts()}')
print(f'Dataset após SMOTE: {y_train_res.value_counts()}')

# 6. Treinamento do Modelo XGBoost
print("Treinando modelo XGBoost...")
start_time_xgb = time.time()
xg_train = xgb.DMatrix(X_train_res, label=y_train_res)
xg_test = xgb.DMatrix(X_test, label=y_test)

params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.01,
    'max_depth': 8,
    'min_child_weight': 60,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': sum(y_train_res == 0) / sum(y_train_res == 1),
    'tree_method': 'hist',
    'verbosity': 0
}

watchlist_xgb = [(xg_train, 'train'), (xg_test, 'valid')]
model_xgb = xgb.train(
    params_xgb,
    xg_train,
    num_boost_round=5000,
    evals=watchlist_xgb,
    early_stopping_rounds=100,
    verbose_eval=500 # Mostrar progresso a cada 500 rodadas
)
print(f"Modelo XGBoost treinado em {time.time() - start_time_xgb:.2f} segundos")
model_xgb.save_model('xgboost_fraud_model_from_mix.json')
print("Modelo XGBoost salvo em 'xgboost_fraud_model_from_mix.json'")

# 7. Treinamento do Modelo LightGBM
print("Treinando modelo LightGBM...")
start_time_lgb = time.time()
lgb_train = lgb.Dataset(X_train_res, y_train_res)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params_lgb = {
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

# Definindo callbacks para logging e early stopping
callbacks_lgb = [
    lgb.log_evaluation(period=500), # Mostrar progresso a cada 500 rodadas
    lgb.early_stopping(stopping_rounds=100)
]

model_lgb = lgb.train(
    params_lgb,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train', 'valid'],
    num_boost_round=5000,
    callbacks=callbacks_lgb
)
print(f"Modelo LightGBM treinado em {time.time() - start_time_lgb:.2f} segundos")
model_lgb.save_model('lightgbm_fraud_model_from_mix.txt')
print("Modelo LightGBM salvo em 'lightgbm_fraud_model_from_mix.txt'")

# 8. Combinação das Predições
print("Combinando predições...")
y_pred_proba_xgb = model_xgb.predict(xg_test, iteration_range=(0, model_xgb.best_iteration + 1))
y_pred_proba_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)

# Média simples das probabilidades
y_pred_proba_combined = (y_pred_proba_xgb + y_pred_proba_lgb) / 2.0

# 9. Avaliação do Modelo Combinado
print("Avaliando modelo combinado...")
average_precision_comb = average_precision_score(y_test, y_pred_proba_combined)
auc_comb = roc_auc_score(y_test, y_pred_proba_combined)
print(f'ROC AUC Combinado: {auc_comb:.4f}')
print(f'Average Precision Combinado: {average_precision_comb:.4f}')

# Encontrar o melhor threshold baseado na curva de precisão-recall para o combinado
precision_comb, recall_comb, thresholds_comb = precision_recall_curve(y_test, y_pred_proba_combined)
f1_scores_comb = 2 * (precision_comb * recall_comb) / (precision_comb + recall_comb + 1e-10) # Adicionado epsilon para evitar divisão por zero
optimal_idx_comb = np.argmax(f1_scores_comb)
# Garantir que optimal_idx_comb esteja dentro dos limites de thresholds_comb
if optimal_idx_comb >= len(thresholds_comb):
    optimal_idx_comb = len(thresholds_comb) - 1
optimal_threshold_comb = thresholds_comb[optimal_idx_comb]
print(f'Threshold ótimo combinado baseado em F1: {optimal_threshold_comb:.4f}')

# Resultados com threshold otimizado
y_pred_optimal_comb = (y_pred_proba_combined >= optimal_threshold_comb).astype(int)
cm_optimal_comb = confusion_matrix(y_test, y_pred_optimal_comb)

print("Resultados combinados com threshold otimizado:")
print(cm_optimal_comb)
print(classification_report(y_test, y_pred_optimal_comb, digits=4))

# Métricas específicas para detecção de fraude
tn_comb, fp_comb, fn_comb, tp_comb = cm_optimal_comb.ravel()
fraud_detection_rate_comb = tp_comb / (tp_comb + fn_comb) if (tp_comb + fn_comb) > 0 else 0
false_alarm_rate_comb = fp_comb / (fp_comb + tn_comb) if (fp_comb + tn_comb) > 0 else 0
print(f"Taxa de detecção de fraude combinada: {fraud_detection_rate_comb:.4f}")
print(f"Taxa de falsos alarmes combinada: {false_alarm_rate_comb:.4f}")

# 10. Visualizações do Modelo Combinado
print("Gerando visualizações...")
plt.figure(figsize=(18, 6)) # Ajustado para 3 gráficos

# 1. Matriz de confusão combinada
plt.subplot(1, 3, 1)
sns.heatmap(cm_optimal_comb, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusão Combinada (Th={optimal_threshold_comb:.3f})')
plt.xlabel('Predito')
plt.ylabel('Real')

# 2. Curva ROC combinada
plt.subplot(1, 3, 2)
fpr_comb, tpr_comb, _ = roc_curve(y_test, y_pred_proba_combined)
plt.plot(fpr_comb, tpr_comb, label=f'AUC Combinado = {auc_comb:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC Combinada')
plt.legend()

# 3. Curva Precision-Recall combinada
plt.subplot(1, 3, 3)
plt.plot(recall_comb, precision_comb, label=f'AP Combinado = {average_precision_comb:.3f}')
# Marcar o threshold ótimo
plt.scatter(recall_comb[optimal_idx_comb], precision_comb[optimal_idx_comb], marker='o', color='red', label=f'Optimal Threshold (F1={f1_scores_comb[optimal_idx_comb]:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall Combinada')
plt.legend()

plt.tight_layout()
plt.savefig('combined_xgb_lgb_results.png')
print("Gráficos salvos em 'combined_xgb_lgb_results.png'")
# plt.show() # Comente ou descomente se quiser exibir interativamente

# 11. Análise de erros do Modelo Combinado
false_positives_comb = X_test[(y_pred_optimal_comb == 1) & (y_test == 0)]
false_negatives_comb = X_test[(y_pred_optimal_comb == 0) & (y_test == 1)]

print(f"Número de falsos positivos (Combinado): {len(false_positives_comb)}")
print(f"Número de falsos negativos (Combinado): {len(false_negatives_comb)}")

print("Script concluído.")
