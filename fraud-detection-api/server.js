const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));  // Aumentando o limite para transações grandes

// Verificar se o modelo existe
const modelPath = path.join(__dirname, '..', 'fraud_model.pt');
const featuresPath = path.join(__dirname, '..', 'feature_columns.pkl');
const thresholdPath = path.join(__dirname, '..', 'threshold.pkl');

// Root endpoint
app.get('/', (req, res) => {
  // Verificar se os arquivos do modelo existem
  const modelExists = fs.existsSync(modelPath);
  const featuresExist = fs.existsSync(featuresPath);
  const thresholdExists = fs.existsSync(thresholdPath);
  
  res.json({
    status: 'Fraud Detection API is running',
    model: {
      path: modelPath,
      exists: modelExists
    },
    features: {
      path: featuresPath,
      exists: featuresExist
    },
    threshold: {
      path: thresholdPath,
      exists: thresholdExists
    }
  });
});

// Prediction endpoint
app.post('/api/predict', (req, res) => {
  console.log('Recebida requisição de predição');
  const transaction = req.body;
  
  if (!transaction) {
    console.error('Dados de transação não fornecidos');
    return res.status(400).json({ error: 'Transaction data is required' });
  }
  
  console.log(`Processando transação ID: ${transaction.TransactionID || transaction.transaction_id || 'unknown'}`);
  
  // Call Python script to make prediction
  const pythonProcess = spawn('python', [
    path.join(__dirname, 'predict.py'),
    JSON.stringify(transaction)
  ]);
  
  let predictionResult = '';
  let errorOutput = '';
  
  // Collect data from script
  pythonProcess.stdout.on('data', (data) => {
    predictionResult += data.toString();
  });
  
  pythonProcess.stderr.on('data', (data) => {
    errorOutput += data.toString();
    console.error(`Erro no script Python: ${data.toString()}`);
  });
  
  // When the script closes
  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Python process exited with code ${code}`);
      console.error(`Error details: ${errorOutput}`);
      return res.status(500).json({ 
        error: 'Error making prediction', 
        details: errorOutput,
        code: code
      });
    }
    
    try {
      const result = JSON.parse(predictionResult);
      
      // Log the prediction result
      if (result.prediction) {
        console.log(`Prediction result: Fraud=${result.prediction.is_fraud}, Probability=${result.prediction.fraud_probability}`);
      } else if (result.error) {
        console.error(`Prediction error: ${result.error}`);
      }
      
      return res.json(result);
    } catch (error) {
      console.error('Error parsing prediction result:', error);
      console.error('Raw output:', predictionResult);
      return res.status(500).json({ 
        error: 'Error parsing prediction result',
        message: error.message,
        raw_output: predictionResult 
      });
    }
  });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  // Verificar se o modelo existe
  const modelExists = fs.existsSync(modelPath);
  const featuresExist = fs.existsSync(featuresPath);
  const thresholdExists = fs.existsSync(thresholdPath);
  
  if (!modelExists || !featuresExist || !thresholdExists) {
    return res.status(503).json({
      status: 'ERROR',
      message: 'Model files not found',
      details: {
        model: modelExists ? 'OK' : 'Missing',
        features: featuresExist ? 'OK' : 'Missing',
        threshold: thresholdExists ? 'OK' : 'Missing'
      }
    });
  }
  
  res.json({ 
    status: 'OK',
    model: 'fraud_model.pt',
    server_time: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Servidor de Detecção de Fraudes rodando na porta ${PORT}`);
  console.log(`Buscando modelo em: ${modelPath}`);
  console.log(`Buscando features em: ${featuresPath}`);
  console.log(`Buscando threshold em: ${thresholdPath}`);
}); 