import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, classification_report
import os
import warnings
import gc
import torch
import pickle
warnings.filterwarnings('ignore')

# Definindo o caminho para os dados
DATA_PATH = 'ieee-fraud-detection'

# Definir uma quantidade reduzida de dados para teste
# Vamos usar 40% dos dados, mas garantindo a mesma proporção de fraudes
SAMPLE_FRAC = 0.40  # 40% dos dados
RANDOM_STATE = 42

# Classe para encapsular o modelo XGBoost em um formato PyTorch
class XGBoostPytorchWrapper(torch.nn.Module):
    def __init__(self, xgb_model, threshold=0.5):
        super(XGBoostPytorchWrapper, self).__init__()
        self.xgb_model = xgb_model
        self.threshold = threshold
        
    def forward(self, x):
        # Converter para formato numpy se for tensor
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
            
        # Converter para DMatrix do XGBoost
        dmat = xgb.DMatrix(x_np)
        
        # Fazer predição
        preds = self.xgb_model.predict(dmat)
        
        # Converter para tensor PyTorch
        return torch.FloatTensor(preds)
    
    # Método para classificação binária
    def predict(self, x):
        probs = self.forward(x)
        return (probs >= self.threshold).float()

# Carregando apenas um amostra inicial para ver a distribuição de fraudes
print('Carregando amostra inicial...')
sample_df = pd.read_csv(os.path.join(DATA_PATH, 'train_transaction.csv'), 
                         usecols=['TransactionID', 'isFraud'], nrows=100000)
print(f'Distribuição de fraudes na amostra: \n{sample_df["isFraud"].value_counts(normalize=True) * 100}%')

# Carregar os dados em chunks mais pequenos
print('Carregando dados de transações...')
dtypes = {}
for i in range(1, 340):  # Para colunas V1-V339
    dtypes[f'V{i}'] = 'float32'

# Carregar apenas um pedaço inicial para análise
train_sample = pd.read_csv(os.path.join(DATA_PATH, 'train_transaction.csv'), nrows=int(590540 * SAMPLE_FRAC))
print(f'Tamanho da amostra carregada: {len(train_sample)} transações')

# Para garantir representação adequada de fraudes
fraud_indices = train_sample[train_sample['isFraud'] == 1].index
non_fraud_indices = train_sample[train_sample['isFraud'] == 0].index

# Verificando quantos casos de fraude temos
n_frauds = len(fraud_indices)
print(f'Número de casos de fraude na amostra: {n_frauds}')

# Se houver muito poucos casos de fraude, podemos aumentar a amostra de fraudes
if n_frauds < 1000:  # Threshold arbitrário, ajuste conforme necessário
    print("Poucos casos de fraude na amostra, carregando mais dados...")
    # Carregamos mais dados para garantir casos de fraude suficientes
    additional_data = pd.read_csv(os.path.join(DATA_PATH, 'train_transaction.csv'), 
                                 skiprows=range(1, len(train_sample) + 1),
                                 nrows=100000)  # Carrega mais 100k linhas
    additional_frauds = additional_data[additional_data['isFraud'] == 1]
    print(f'Casos adicionais de fraude encontrados: {len(additional_frauds)}')
    
    # Adicionar casos de fraude à amostra original
    train_sample = pd.concat([train_sample, additional_frauds])
    print(f'Novo tamanho da amostra: {len(train_sample)}')

# Vamos fazer undersampling nas transações não-fraude para melhorar o balanceamento
fraud_ratio = len(fraud_indices) / len(train_sample)
print(f'Proporção original de fraudes: {fraud_ratio:.4f}')

# Definir uma proporção alvo mais equilibrada (ex: 1:3 entre fraudes e não-fraudes)
target_ratio = 0.25  # 25% de fraudes
if fraud_ratio < target_ratio:
    # Calcular quantos casos não-fraude precisamos para a proporção alvo
    n_non_fraud_target = int(n_frauds / target_ratio) - n_frauds
    if n_non_fraud_target < len(non_fraud_indices):
        # Selecionar aleatoriamente casos não-fraude
        non_fraud_sample = np.random.choice(non_fraud_indices, n_non_fraud_target, replace=False)
        # Criar um novo dataframe mais balanceado
        balanced_indices = np.concatenate([fraud_indices, non_fraud_sample])
        train_sample = train_sample.loc[balanced_indices]
        print(f'Dataframe balanceado: {len(train_sample)} transações')
        print(f'Nova proporção de fraudes: {len(fraud_indices) / len(train_sample):.4f}')

# Carregar dados de identidade para a amostra
print('Carregando dados de identidade...')
train_identity = pd.read_csv(os.path.join(DATA_PATH, 'train_identity.csv'))

# Filtrando apenas IDs que estão na amostra
train_identity = train_identity[train_identity['TransactionID'].isin(train_sample['TransactionID'])]

# Carregar dados de teste (também reduzidos)
print('Carregando dados de teste...')
test_transaction = pd.read_csv(os.path.join(DATA_PATH, 'test_transaction.csv'), 
                              nrows=int(506691 * SAMPLE_FRAC))  # Mesma proporção para teste

test_identity = pd.read_csv(os.path.join(DATA_PATH, 'test_identity.csv'))
test_identity = test_identity[test_identity['TransactionID'].isin(test_transaction['TransactionID'])]

# Fusão dos datasets
print('Combinando datasets...')
train = train_sample.merge(train_identity, on='TransactionID', how='left')
test = test_transaction.merge(test_identity, on='TransactionID', how='left')

# Liberar memória
del train_sample, train_identity, test_transaction, test_identity
gc.collect()

# Verificar distribuição final de fraudes
print(f'Distribuição final de fraudes: \n{train["isFraud"].value_counts()}')
print(f'Porcentagem de fraudes: {train["isFraud"].mean() * 100:.2f}%')

# Separando o target e IDs
y = train['isFraud']
train_id = train['TransactionID']
test_id = test['TransactionID']

# Removendo features desnecessárias
del train['isFraud'], train['TransactionID'], test['TransactionID']

# Preparando as features categóricas
cat_features = [col for col in train.columns if train[col].dtype == 'object']
print(f'Total de features categóricas: {len(cat_features)}')

# Encontrar colunas comuns entre train e test
common_features = set(train.columns).intersection(set(test.columns))
print(f'Total de features em comum: {len(common_features)}')

# Processando features categóricas - apenas as que existem em ambos os conjuntos
for feature in cat_features:
    if feature in common_features:
        print(f'Processando feature categórica: {feature}')
        le = LabelEncoder()
        # Combinando train e test para garantir todas as categorias
        le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
        train[feature] = le.transform(list(train[feature].astype(str).values))
        test[feature] = le.transform(list(test[feature].astype(str).values))
    else:
        print(f'Feature {feature} não existe no conjunto de teste, ignorando...')
        # Converter para numérica usando apenas o conjunto de treino
        le = LabelEncoder()
        le.fit(list(train[feature].astype(str).values))
        train[feature] = le.transform(list(train[feature].astype(str).values))

# Garantir que todas as colunas no teste estejam presentes no treino
for feature in list(test.columns):
    if feature not in train.columns:
        print(f'Feature {feature} existe apenas no conjunto de teste, removendo...')
        test = test.drop(columns=[feature])

# Preenchendo valores nulos
train = train.fillna(-999)
test = test.fillna(-999)

# Salvar as colunas para uso posterior na inferência
feature_columns = list(train.columns)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print(f"Colunas salvas em 'feature_columns.pkl' para uso posterior")

# Calcular peso para cada classe para lidar com desbalanceamento
fraud_weight = (1 - y.mean()) / y.mean()  # Quanto maior o peso, mais importância para a classe minoritária
print(f'Peso calculado para classe fraude: {fraud_weight:.2f}')

# Configuração do modelo XGBoost com ajustes para melhorar a detecção de fraudes
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 5,  # Aumentado para capturar relações mais complexas
    'n_estimators': 500,  # Aumentado para melhorar performance
    'subsample': 0.9,
    'colsample_bytree': 0.8,  # Ajustado para reduzir overfitting
    'tree_method': 'hist',  # Usa o algoritmo histograma que é mais eficiente em memória
    'max_bin': 256,  # Reduz o número de bins para acelerar o treinamento
    'scale_pos_weight': fraud_weight  # Ajusta o peso da classe minoritária
}

# Validação cruzada e treinamento
n_fold = 5  # Reduzido para economizar tempo
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()

print('Iniciando treinamento com validação cruzada...')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, y)):
    print(f'Fold {fold_+1}')
    
    # Converter para DMatrix
    print("Convertendo dados de treino para DMatrix...")
    trn_data = xgb.DMatrix(train.iloc[trn_idx], label=y.iloc[trn_idx])
    
    print("Convertendo dados de validação para DMatrix...")
    val_data = xgb.DMatrix(train.iloc[val_idx], label=y.iloc[val_idx])
    
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    
    print("Treinando modelo...")
    clf = xgb.train(params, trn_data, 500, watchlist, 
                   early_stopping_rounds=50, verbose_eval=50)
    
    print("Fazendo previsões no conjunto de validação...")
    oof[val_idx] = clf.predict(val_data)
    
    # Importância das features
    print("Calculando importância das features...")
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = list(train.columns)
    feature_imp = clf.get_score(importance_type='gain')
    # Verificar features que não aparecem no get_score pois não foram usadas
    for col in train.columns:
        if col not in feature_imp:
            feature_imp[col] = 0
    fold_importance["importance"] = [feature_imp.get(col, 0) for col in train.columns]
    fold_importance["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
    
    # Predição
    print("Fazendo previsões no conjunto de teste...")
    # Garantir que o teste tenha exatamente as mesmas colunas que o treino, na mesma ordem
    missing_cols = set(train.columns) - set(test.columns)
    for col in missing_cols:
        test[col] = -999  # Preencher colunas ausentes com valor padrão
    
    # Garantir a mesma ordem das colunas
    test = test[train.columns]
    
    dtest = xgb.DMatrix(test)
    predictions += clf.predict(dtest) / n_fold
    
    # Liberar memória
    del trn_data, val_data
    gc.collect()

print(f'AUC geral: {roc_auc_score(y, oof)}')

# Encontrar o melhor threshold de classificação usando a curva precision-recall
print("Otimizando threshold usando curva precision-recall...")
precision, recall, thresholds = precision_recall_curve(y, oof)

# Calcular F1 para cada threshold
f1_scores = 2 * precision * recall / (precision + recall + 1e-7)  # Evitar divisão por zero
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print(f'Melhor threshold encontrado: {best_threshold:.4f}')
print(f'F1 Score com este threshold: {f1_scores[best_threshold_idx]:.4f}')
print(f'Precision com este threshold: {precision[best_threshold_idx]:.4f}')
print(f'Recall com este threshold: {recall[best_threshold_idx]:.4f}')

# Aplicar o threshold otimizado e gerar as previsões finais
y_pred_binary = (oof >= best_threshold).astype(int)

# Mostrar relatório de classificação detalhado
print("\nRelatório de classificação com threshold otimizado:")
print(classification_report(y, y_pred_binary))

# Visualizar as features mais importantes
if len(feature_importance_df) > 0:
    print("Gerando gráfico de importância das features...")
    cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:20].index

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(14, 10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Top 20 Features (média entre folds)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Gráfico salvo como 'feature_importance.png'")

# Salvando o modelo final
print("Treinando modelo final em todos os dados...")
dmatrix_full = xgb.DMatrix(train, label=y)
final_model = xgb.train(params, dmatrix_full, 500)

# Salvando o modelo em formato nativo do XGBoost
final_model.save_model('xgboost_fraud_model.json')
print('Modelo XGBoost salvo como "xgboost_fraud_model.json"')

# Criando wrapper PyTorch para o modelo
print("Criando wrapper PyTorch para o modelo XGBoost...")
pytorch_model = XGBoostPytorchWrapper(final_model, threshold=best_threshold)

# Salvar também o threshold otimizado
threshold_dict = {'threshold': best_threshold}
with open('threshold.pkl', 'wb') as f:
    pickle.dump(threshold_dict, f)
print(f"Threshold otimizado ({best_threshold:.4f}) salvo como 'threshold.pkl'")

# Salvar o modelo em formato PyTorch
torch.save(pytorch_model, 'fraud_model.pt')
print('Modelo PyTorch salvo como "fraud_model.pt"')

# Criando submission
# Garantir novamente que o teste tenha as mesmas colunas que o treino
missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    test[col] = -999
test = test[train.columns]

final_dtest = xgb.DMatrix(test)
final_predictions = final_model.predict(final_dtest)

submission = pd.DataFrame({
    'TransactionID': test_id,
    'isFraud': final_predictions
})
submission.to_csv('submission.csv', index=False)

# Gerando matriz de confusão com threshold padrão de 0.5
plt.figure(figsize=(10, 8))
conf_matrix_default = confusion_matrix(y, (oof >= 0.5).astype(int))
sns.heatmap(conf_matrix_default, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Não-Fraude', 'Fraude'],
           yticklabels=['Não-Fraude', 'Fraude'])
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão (threshold = 0.5)')
plt.tight_layout()
plt.savefig('confusion_matrix_default.png')
print("Matriz de confusão com threshold padrão salva como 'confusion_matrix_default.png'")

# Gerando matriz de confusão com threshold otimizado
plt.figure(figsize=(10, 8))
conf_matrix_optimized = confusion_matrix(y, y_pred_binary)
sns.heatmap(conf_matrix_optimized, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Não-Fraude', 'Fraude'],
           yticklabels=['Não-Fraude', 'Fraude'])
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title(f'Matriz de Confusão (threshold = {best_threshold:.4f})')
plt.tight_layout()
plt.savefig('confusion_matrix_optimized.png')
print(f"Matriz de confusão com threshold otimizado salva como 'confusion_matrix_optimized.png'")

# Plotar curva Precision-Recall
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, marker='.', label='Curva Precision-Recall')
plt.scatter(recall[best_threshold_idx], precision[best_threshold_idx], marker='o', color='red', 
           label=f'Melhor threshold: {best_threshold:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall_curve.png')
print("Curva Precision-Recall salva como 'precision_recall_curve.png'")


print('Processo concluído. Modelo salvo como "fraud_model.pt" para uso no servidor.')
print('Previsões salvas como "submission.csv"') 

"""
Acurácia: 90.9% (melhorou a detecção geral)
Precisão: 85.3% (quando o modelo prevê fraude, acerta em 85.3% dos casos)
Recall/Sensibilidade: 77.0% (agora detecta 77% de todas as fraudes!)
Especificidade: 95.6% (ainda identifica bem transações legítimas)
F1-Score: 81.0% (bom equilíbrio entre precisão e recall)
Comparando com a anterior:
O recall aumentou drasticamente (de 38.7% para 77.0%)
A quantidade de fraudes detectadas dobrou (de 1505 para 2996)
Houve um pequeno aumento nos falsos positivos, mas é um trade-off necessário
"""