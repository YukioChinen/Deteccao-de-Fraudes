import sys
import json
import pandas as pd
import numpy as np
import torch
import pickle
import xgboost as xgb
import os

# Definir a mesma classe que foi usada para salvar o modelo
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

def predict_fraud(transaction_data):
    try:
        # Paths to model files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'fraud_model.pt')
        feature_columns_path = os.path.join(base_dir, 'feature_columns.pkl')
        threshold_path = os.path.join(base_dir, 'threshold.pkl')
        
        # Load the model com parâmetro weights_only=False para permitir carregar classes personalizadas
        model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
        
        # Load feature columns
        with open(feature_columns_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Load threshold (opcional, já que está embutido no modelo)
        with open(threshold_path, 'rb') as f:
            threshold_dict = pickle.load(f)
            optimal_threshold = threshold_dict['threshold']
        
        # Create a dataframe from the transaction
        df = pd.DataFrame([transaction_data])
        
        # Converter colunas específicas para strings (campos que sabemos que são categóricos)
        categorical_cols = ['ProductCD', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Process the dataframe
        # Ensure all required columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = -999  # Default value for missing columns
        
        # Ensure correct column order
        df = df[feature_columns]
        
        # Fill missing values
        df = df.fillna(-999)
        
        # Converter todas as colunas categóricas para numéricas
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                # Mapear valores de string para inteiros
                unique_values = df[col].unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                df[col] = df[col].map(value_map).fillna(-999).astype('float32')
        
        # Converter todos os valores para float32 para garantir compatibilidade
        for col in df.columns:
            df[col] = df[col].astype('float32')
        
        # Make prediction
        with torch.no_grad():
            prediction_tensor = model(df)
            # Garantir que temos um valor Python nativo
            raw_prediction = float(prediction_tensor.numpy()[0])
        
        # Determine if it's fraud based on threshold
        is_fraud = bool(raw_prediction >= optimal_threshold)
        
        # Formatar o resultado com tipos Python nativos para garantir serialização JSON
        result = {
            'prediction': {
                'is_fraud': is_fraud,
                'fraud_probability': float(raw_prediction),
                'threshold': float(optimal_threshold)
            },
            'transaction_id': str(transaction_data.get('TransactionID', transaction_data.get('transaction_id', 'unknown')))
        }
        
        return result
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERRO: {str(e)}")
        print(f"TRACEBACK: {error_traceback}")
        return {
            'error': str(e),
            'traceback': error_traceback
        }

if __name__ == '__main__':
    try:
        # Read transaction data from command line argument
        transaction_json = sys.argv[1]
        transaction_data = json.loads(transaction_json)
        
        # Make prediction
        result = predict_fraud(transaction_data)
        
        # Garantir que o resultado é serializável
        json_result = json.dumps(result)
        
        # Return result as JSON
        print(json_result)
    except Exception as e:
        import traceback
        error_result = {
            'error': f"Erro no script principal: {str(e)}",
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_result)) 