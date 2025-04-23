# XGBoost com Validação Cruzada e SMOTE

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
import pandas as pd
import os

df = pd.read_csv('creditcard.csv')

# Preparar dados
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Validação cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_true_all = []
y_pred_proba_all = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nTreinando fold {fold}...")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.01,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),
        'tree_method': 'hist',
        'verbosity': 0
    }

    model = xgb.train(params, dtrain, num_boost_round=500)

    y_pred_proba = model.predict(dtest)

    y_true_all.extend(y_test)
    y_pred_proba_all.extend(y_pred_proba)

# Conversão para array numpy
y_true_all = np.array(y_true_all)
y_pred_proba_all = np.array(y_pred_proba_all)

# Calcular métricas finais
auc = roc_auc_score(y_true_all, y_pred_proba_all)
ap = average_precision_score(y_true_all, y_pred_proba_all)
precision, recall, thresholds = precision_recall_curve(y_true_all, y_pred_proba_all)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nROC AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print(f"Threshold ótimo baseado em F1: {optimal_threshold:.4f}")

# Predição com threshold ótimo
y_pred_optimal = (y_pred_proba_all >= optimal_threshold).astype(int)
cm = confusion_matrix(y_true_all, y_pred_optimal)

# Relatório
print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_optimal))

# Taxas específicas
tn, fp, fn, tp = cm.ravel()
print(f"Taxa de detecção de fraude: {tp / (tp + fn):.4f}")
print(f"Taxa de falsos alarmes: {fp / (fp + tn):.4f}")

# ====== Gráficos ======
plt.figure(figsize=(12, 10))

# 1. Matriz de confusão
plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusão (threshold={optimal_threshold:.3f})')
plt.xlabel('Predito')
plt.ylabel('Real')

# 2. Curva ROC
plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_true_all, y_pred_proba_all)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()

# 3. Curva Precision-Recall
plt.subplot(2, 2, 3)
plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.axvline(recall[optimal_idx], color='r', linestyle='--',
            label=f'Threshold: {optimal_threshold:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()

# 4. Importância das features
plt.subplot(2, 2, 4)
importance_dict = model.get_score(importance_type='weight')
features = list(importance_dict.keys())
importances = list(importance_dict.values())
sorted_idx = np.argsort(importances)[-10:]  # Top 10

plt.barh(range(len(sorted_idx)), np.array(importances)[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
plt.title('Importância das Features (Top 10)')

# Criar diretório de saída
output_dir = 'XGBoost'
os.makedirs(output_dir, exist_ok=True)

# Salvar figura
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'xgboost_crossvalidation_results.png'))
plt.show()


# Salvar o modelo
model.save_model(os.path.join(output_dir, 'xgboost_crossvalidation_model.json'))
print("Modelo salvo como 'xgboost_crossvalidation_model.json'")