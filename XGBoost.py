import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
data = pd.read_csv('creditcard.csv')

# 2. Análise exploratória simples
print(data.info())
print(data['Class'].value_counts())  # Verificando o desbalanceamento

# 3. Separar features e target
X = data.drop(['Class', 'Time'], axis=1)  # Removendo 'Time' que não contribui muito
y = data['Class']

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Balanceamento dos dados usando SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f'Dataset original: {y_train.value_counts()}')
print(f'Dataset balanceado: {y_train_res.value_counts()}')

# 6. Treinamento com XGBoost
xg_train = xgb.DMatrix(X_train_res, label=y_train_res)
xg_test = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.05,
    'max_depth': 6,
    'verbosity': 0,
    'early_stopping_rounds': 50
}

watchlist = [(xg_train, 'train'), (xg_test, 'test')]

model = xgb.train(params, xg_train, num_boost_round=1000, evals=watchlist)

# 7. Avaliação do modelo
y_pred_proba = model.predict(xg_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Métricas
print(f'ROC AUC: {roc_auc_score(y_test, y_pred_proba)}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Matriz de confusão visual
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.show()

# 8. Importância das features
xgb.plot_importance(model)
plt.show()