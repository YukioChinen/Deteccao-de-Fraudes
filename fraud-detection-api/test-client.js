const fetch = require('node-fetch');

// ============================================================
// CASOS DE TESTE COM RESULTADOS ESPERADOS
// ============================================================

// CASO 1: FRAUDE - Transação de alto valor com comportamento suspeito
// Resultado esperado: FRAUDE
// Motivo: Valor alto, e-mail diferente, combinação de flags suspeitas
const fraudCase1 = {
  "TransactionID": "fraud-test-001",
  "TransactionDT": 86400,
  "TransactionAmt": 95000.00, // Valor alto
  "ProductCD": "H", // Produto de alto risco
  "card1": 9283,
  "card2": 321,
  "card3": 150,
  "card4": 222,
  "card5": 100,
  "card6": "credit",
  "addr1": 123,
  "addr2": 456,
  "dist1": 50, // Distância aumentada
  "dist2": 20,
  "P_emaildomain": "temp-mail.org", // Domínio temporário de alto risco
  "R_emaildomain": "gmail.com", // Mismatch de domínios
  "C1": 1.0,
  "C2": 1.0,
  "C3": 0.0,
  "C4": 0.0,
  "C5": 0.0,
  "C6": 1.0,
  "C7": 1.0,
  "C8": 0.0,
  "C9": 1.0, // Sinal de problema
  "C10": 1.0, // Sinal de problema
  "C11": 1.0,
  "C12": 0.0,
  "C13": 1.0,
  "C14": 1.0,
  "D1": 0.0, // Padrão temporal inconsistente
  "D2": 0.0,
  "D3": 0.0,
  "D4": 1.0,
  "D5": 0.0,
  "D6": 0.0,
  "D7": 0.0,
  "D8": 1.0,
  "D9": 0.0,
  "D10": 1.0,
  "D11": 1.0,
  "D12": 0.0,
  "D13": 0.0,
  "D14": 0.0,
  "D15": 1.0,
  "M1": "T",
  "M2": "F", // Flags inconsistentes
  "M3": "T",
  "M4": "M1",
  "M5": "T",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

// CASO 2: FRAUDE - Transação com comportamento inconsistente
// Resultado esperado: FRAUDE
// Motivo: Múltiplos sinais de inconsistência entre dados de identidade
const fraudCase2 = {
  "TransactionID": "fraud-test-002",
  "TransactionDT": 125000,
  "TransactionAmt": 499.99,
  "ProductCD": "W", // Produto de risco médio
  "card1": 3845,
  "card2": 567,
  "card3": 243,
  "card4": 111,
  "card5": 142,
  "card6": "debit",
  "addr1": 733,
  "addr2": 0, // Endereço incompleto - suspeito
  "dist1": 324,
  "dist2": 0, // Distância inconsistente - suspeito
  "P_emaildomain": "protonmail.com", // E-mail anônimo - suspeito
  "R_emaildomain": "hotmail.com", // Mismatch de domínios
  "C1": 2.0, // Valor anômalo
  "C2": 2.0, // Valor anômalo
  "C3": 0.0,
  "C4": 1.0,
  "C5": 0.0,
  "C6": 0.0,
  "C7": 0.0,
  "C8": 1.0,
  "C9": 1.0, // Sinal de problema
  "C10": 0.0,
  "C11": 1.0,
  "C12": 1.0, // Sinal de problema
  "C13": 1.0,
  "C14": 0.0,
  "D1": 0.0,
  "D2": 0.0,
  "D3": 0.0,
  "D4": 0.0,
  "D5": 0.0,
  "D6": 0.0,
  "D7": 0.0,
  "D8": 0.0,
  "D9": 0.0,
  "D10": 0.0,
  "D11": 1.0,
  "D12": 0.0,
  "D13": 0.0,
  "D14": 0.0,
  "D15": 1.0,
  "M1": "T",
  "M2": "F",
  "M3": "F", // Conjunto inconsistente de flags
  "M4": "M2",
  "M5": "F",
  "M6": "F", // Conjunto inconsistente de flags
  "M7": "T",
  "M8": "F",
  "M9": "T"
};

// CASO 3: FRAUDE - Transação com padrão de tempo suspeito
// Resultado esperado: FRAUDE
// Motivo: Padrão de tempo anômalo, alto valor, múltiplos sinais
const fraudCase3 = {
  "TransactionID": "fraud-test-003",
  "TransactionDT": 3600, // Hora incomum (1h da manhã)
  "TransactionAmt": 1299.99,
  "ProductCD": "H", // Produto de alto risco
  "card1": 5498,
  "card2": 888,
  "card3": 123,
  "card4": 456,
  "card5": 789,
  "card6": "credit",
  "addr1": 444,
  "addr2": 333,
  "dist1": 500, // Distância grande
  "dist2": 500, // Distância grande
  "P_emaildomain": "yahoo.com",
  "R_emaildomain": "yahoo.co.jp", // Domínio internacional diferente
  "C1": 4.0, // Valor muito anômalo
  "C2": 1.0,
  "C3": 0.0,
  "C4": 0.0,
  "C5": 0.0,
  "C6": 1.0,
  "C7": 1.0,
  "C8": 0.0,
  "C9": 0.0,
  "C10": 0.0,
  "C11": 0.0,
  "C12": 1.0,
  "C13": 1.0,
  "C14": 1.0,
  "D1": 3.0, // Padrão de tempo anômalo
  "D2": 0.0,
  "D3": 0.0,
  "D4": 0.0,
  "D5": 0.0,
  "D6": 0.0,
  "D7": 0.0,
  "D8": 0.0,
  "D9": 0.0,
  "D10": 1.0,
  "D11": 1.0,
  "D12": 0.0,
  "D13": 0.0,
  "D14": 0.0,
  "D15": 1.0,
  "M1": "F", // Muitas flags inconsistentes
  "M2": "T",
  "M3": "F",
  "M4": "M0",
  "M5": "F",
  "M6": "F",
  "M7": "F",
  "M8": "T",
  "M9": "T"
};

// CASO 4: NÃO-FRAUDE - Transação de baixo valor com comportamento normal
// Resultado esperado: NÃO-FRAUDE
// Motivo: Valor baixo, comportamento consistente, padrões normais
const nonFraudCase1 = {
  "TransactionID": "legit-test-001",
  "TransactionDT": 43200, // Meio-dia - horário normal de compra
  "TransactionAmt": 25.50, // Valor baixo
  "ProductCD": "C", // Produto de baixo risco
  "card1": 1234,
  "card2": 123,
  "card3": 456,
  "card4": 789,
  "card5": 100,
  "card6": "debit",
  "addr1": 123,
  "addr2": 456,
  "dist1": 10,
  "dist2": 10, // Distâncias consistentes
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com", // Mesmo domínio de e-mail - consistente
  "C1": 1.0,
  "C2": 1.0,
  "C3": 0.0,
  "C4": 0.0,
  "C5": 0.0,
  "C6": 0.0,
  "C7": 0.0,
  "C8": 0.0,
  "C9": 0.0, // Sem sinais de problema
  "C10": 0.0, // Sem sinais de problema
  "C11": 0.0,
  "C12": 0.0,
  "C13": 0.0,
  "C14": 0.0,
  "D1": 1.0, // Padrão temporal normal
  "D2": 1.0,
  "D3": 0.0,
  "D4": 0.0,
  "D5": 0.0,
  "D6": 0.0,
  "D7": 0.0,
  "D8": 0.0,
  "D9": 0.0,
  "D10": 0.0,
  "D11": 0.0,
  "D12": 0.0,
  "D13": 0.0,
  "D14": 0.0,
  "D15": 0.0,
  "M1": "T", // Padrão consistente de flags
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

// CASO 5: NÃO-FRAUDE - Transação recorrente
// Resultado esperado: NÃO-FRAUDE
// Motivo: Padrão típico de assinatura mensal
const nonFraudCase2 = {
  "TransactionID": "legit-test-002",
  "TransactionDT": 86400, // Período do dia consistente
  "TransactionAmt": 9.99, // Valor típico de assinatura
  "ProductCD": "R", // Produto recorrente
  "card1": 5678,
  "card2": 234,
  "card3": 567,
  "card4": 890,
  "card5": 100,
  "card6": "credit",
  "addr1": 789,
  "addr2": 12, // Corrigido número octal 012 para decimal 12
  "dist1": 5,
  "dist2": 5, // Distâncias consistentes e pequenas
  "P_emaildomain": "outlook.com",
  "R_emaildomain": "outlook.com", // Mesmo domínio
  "C1": 1.0,
  "C2": 1.0,
  "C3": 1.0, // Indicador de recorrência
  "C4": 1.0, // Indicador de recorrência
  "C5": 1.0, // Indicador de recorrência
  "C6": 0.0,
  "C7": 0.0,
  "C8": 0.0,
  "C9": 0.0,
  "C10": 0.0,
  "C11": 0.0,
  "C12": 0.0,
  "C13": 0.0,
  "C14": 0.0,
  "D1": 30.0, // ~30 dias desde última transação (mensal)
  "D2": 30.0, // Consistente com D1
  "D3": 30.0, // Consistente com padrão mensal
  "D4": 30.0, // Consistente com padrão mensal
  "D5": 0.0,
  "D6": 0.0,
  "D7": 0.0,
  "D8": 0.0,
  "D9": 0.0,
  "D10": 0.0,
  "D11": 0.0,
  "D12": 0.0,
  "D13": 0.0,
  "D14": 0.0,
  "D15": 0.0,
  "M1": "T", // Padrão consistente
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

// CASO 6: NÃO-FRAUDE - Transação com cliente frequente
// Resultado esperado: NÃO-FRAUDE
// Motivo: Cliente frequente, comportamento consistente
const nonFraudCase3 = {
  "TransactionID": "legit-test-003",
  "TransactionDT": 57600, // Horário normal de trabalho (16h)
  "TransactionAmt": 149.95, // Valor médio
  "ProductCD": "C", // Baixo risco - Mudado para C abaixo para teste
  "card1": 4321,
  "card2": 432,
  "card3": 765,
  "card4": 198,
  "card5": 100,
  "card6": "debit",
  "addr1": 555,
  "addr2": 444,
  "dist1": 15,
  "dist2": 15, // Consistente
  "P_emaildomain": "yahoo.com",
  "R_emaildomain": "yahoo.com", // Mesmo domínio
  "C1": 1.0,
  "C2": 1.0,
  "C3": 0.0,
  "C4": 0.0,
  "C5": 0.0,
  "C6": 0.0,
  "C7": 0.0,
  "C8": 0.0,
  "C9": 0.0,
  "C10": 0.0,
  "C11": 0.0,
  "C12": 0.0,
  "C13": 1.0, // Cliente frequente
  "C14": 1.0, // Cliente frequente
  "D1": 7.0, // ~7 dias desde última transação (semanal)
  "D2": 7.0, // Consistente com D1
  "D3": 7.0, // Consistente com padrão semanal
  "D4": 0.0,
  "D5": 0.0,
  "D6": 0.0,
  "D7": 0.0,
  "D8": 0.0,
  "D9": 0.0,
  "D10": 0.0,
  "D11": 0.0,
  "D12": 0.0,
  "D13": 0.0,
  "D14": 0.0,
  "D15": 0.0,
  "M1": "T", // Padrão consistente
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

// CASOS DE FRAUDE ADICIONAIS
const fraudCase4 = {
  "TransactionID": "fraud-test-004",
  "TransactionDT": 7200, // 2 AM - hora suspeita
  "TransactionAmt": 2500.00, // Valor alto
  "ProductCD": "H", // Produto de alto risco
  "card1": 7845,
  "card2": 321,
  "card3": 234,
  "card4": 567,
  "card5": 234,
  "card6": "credit",
  "addr1": 452,
  "addr2": 0, // Endereço incompleto
  "dist1": 700, // Distância grande
  "dist2": 0, // Distância inconsistente
  "P_emaildomain": "yandex.ru", // Domínio de e-mail de risco
  "R_emaildomain": "outlook.com", // Domínios diferentes
  "M1": "F",
  "M2": "T",
  "M3": "F",
  "M4": "M2",
  "M5": "F",
  "M6": "F",
  "M7": "T",
  "M8": "T",
  "M9": "F"
};

const fraudCase5 = {
  "TransactionID": "fraud-test-005",
  "TransactionDT": 10800, // 3 AM - hora suspeita
  "TransactionAmt": 1888.99, // Valor significativo
  "ProductCD": "W", // Produto de risco médio
  "card1": 3333,
  "card2": 444,
  "card3": 555,
  "card4": 666,
  "card5": 777,
  "card6": "debit",
  "addr1": 111,
  "addr2": 222,
  "dist1": 400, // Distância grande
  "dist2": 50, // Distância inconsistente
  "P_emaildomain": "aol.com",
  "R_emaildomain": "protonmail.com", // E-mail anônimo
  "M1": "T",
  "M2": "F",
  "M3": "F",
  "M4": "M1",
  "M5": "T",
  "M6": "F",
  "M7": "F",
  "M8": "F",
  "M9": "T"
};

const fraudCase6 = {
  "TransactionID": "fraud-test-006",
  "TransactionDT": 82800, // 23 PM - hora suspeita
  "TransactionAmt": 4350.25, // Valor muito alto
  "ProductCD": "S", // Mistura 
  "card1": 9999,
  "card2": 888,
  "card3": 777,
  "card4": 666,
  "card5": 555,
  "card6": "credit",
  "addr1": 999,
  "addr2": 888,
  "dist1": 600, // Distância grande
  "dist2": 600,
  "P_emaildomain": "mail.ru", // Domínio de e-mail de risco
  "R_emaildomain": "gmail.com", // Domínios diferentes
  "M1": "F",
  "M2": "F",
  "M3": "F", // Muitos "F" seguidos - inconsistente
  "M4": "M3",
  "M5": "F",
  "M6": "F",
  "M7": "F",
  "M8": "F",
  "M9": "F"
};

const fraudCase7 = {
  "TransactionID": "fraud-test-007",
  "TransactionDT": 14400, // 4 AM
  "TransactionAmt": 999.98, // Valor abaixo de 1000 - evitando limites
  "ProductCD": "H", // Alto risco
  "card1": 1111,
  "card2": 222,
  "card3": 333,
  "card4": 444,
  "card5": 100,
  "card6": "credit",
  "addr1": 777,
  "addr2": 888,
  "dist1": 300,
  "dist2": 10, // Grande disparidade entre distâncias
  "P_emaildomain": "yahoo.co.jp", // Domínio internacional
  "R_emaildomain": "hotmail.com", // Domínios diferentes
  "M1": "T",
  "M2": "T",
  "M3": "F",
  "M4": "M0",
  "M5": "T",
  "M6": "F",
  "M7": "T",
  "M8": "F",
  "M9": "T"
};

const fraudCase8 = {
  "TransactionID": "fraud-test-008",
  "TransactionDT": 43200, // 12 PM - hora comum, mas outros indicadores
  "TransactionAmt": 5000.00, // Valor muito alto
  "ProductCD": "H", // Alto risco
  "card1": 5000,
  "card2": 600,
  "card3": 700,
  "card4": 800,
  "card5": 200,
  "card6": "credit",
  "addr1": 432,
  "addr2": 765,
  "dist1": 800, // Distância muito grande
  "dist2": 800,
  "P_emaildomain": "temp-mail.org", // Domínio temporário - alto risco
  "R_emaildomain": "outlook.com", // Domínios diferentes
  "M1": "F",
  "M2": "T",
  "M3": "F",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "T",
  "M8": "F",
  "M9": "F"
};

const fraudCase9 = {
  "TransactionID": "fraud-test-009",
  "TransactionDT": 18000, // 5 AM
  "TransactionAmt": 2345.67, // Valor alto
  "ProductCD": "W", // Médio risco
  "card1": 4545,
  "card2": 454,
  "card3": 454,
  "card4": 454,
  "card5": 454,
  "card6": "debit",
  "addr1": 454,
  "addr2": 454,
  "dist1": 500,
  "dist2": 100, // Inconsistente
  "P_emaildomain": "mailinator.com", // Serviço de e-mail temporário - alto risco
  "R_emaildomain": "gmail.com", // Domínios diferentes
  "M1": "T",
  "M2": "F",
  "M3": "T",
  "M4": "M2",
  "M5": "T",
  "M6": "F",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

const fraudCase10 = {
  "TransactionID": "fraud-test-010",
  "TransactionDT": 79200, // 22 PM - noite
  "TransactionAmt": 3333.33, // Valor alto com padrão repetitivo
  "ProductCD": "H", // Alto risco
  "card1": 7777,
  "card2": 777,
  "card3": 77,
  "card4": 777,
  "card5": 77,
  "card6": "credit",
  "addr1": 777,
  "addr2": 77,
  "dist1": 777,
  "dist2": 77, // Padrão suspeito de números repetidos
  "P_emaildomain": "sharklasers.com", // Serviço de e-mail temporário - alto risco
  "R_emaildomain": "yahoo.com", // Domínios diferentes
  "M1": "F",
  "M2": "F",
  "M3": "F",
  "M4": "M1",
  "M5": "F",
  "M6": "F",
  "M7": "F",
  "M8": "F",
  "M9": "F"
};

// CASOS DE NÃO-FRAUDE ADICIONAIS
const nonFraudCase4 = {
  "TransactionID": "legit-test-004",
  "TransactionDT": 46800, // 13 PM - horário de almoço
  "TransactionAmt": 19.99, // Valor baixo
  "ProductCD": "C", // Baixo risco
  "card1": 1234,
  "card2": 123,
  "card3": 123,
  "card4": 123,
  "card5": 123,
  "card6": "debit",
  "addr1": 123,
  "addr2": 123,
  "dist1": 5, // Distância pequena
  "dist2": 5, // Consistente
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com", // Mesmo domínio
  "M1": "T",
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

const nonFraudCase5 = {
  "TransactionID": "legit-test-005",
  "TransactionDT": 50400, // 14 PM - horário normal
  "TransactionAmt": 45.50, // Valor baixo
  "ProductCD": "R", // Recorrente
  "card1": 3456,
  "card2": 345,
  "card3": 345,
  "card4": 345,
  "card5": 345,
  "card6": "credit",
  "addr1": 345,
  "addr2": 345,
  "dist1": 3, // Distância pequena
  "dist2": 3, // Consistente
  "P_emaildomain": "hotmail.com",
  "R_emaildomain": "hotmail.com", // Mesmo domínio
  "M1": "T",
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

const nonFraudCase6 = {
  "TransactionID": "legit-test-006",
  "TransactionDT": 54000, // 15 PM - horário normal
  "TransactionAmt": 5.99, // Valor muito baixo
  "ProductCD": "R", // Recorrente
  "card1": 5678,
  "card2": 567,
  "card3": 567,
  "card4": 567,
  "card5": 567,
  "card6": "debit",
  "addr1": 567,
  "addr2": 567,
  "dist1": 1, // Distância muito pequena
  "dist2": 1, // Consistente
  "P_emaildomain": "outlook.com",
  "R_emaildomain": "outlook.com", // Mesmo domínio
  "M1": "T",
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

const nonFraudCase7 = {
  "TransactionID": "legit-test-007",
  "TransactionDT": 57600, // 16 PM - horário normal
  "TransactionAmt": 29.99, // Valor baixo
  "ProductCD": "S", // Baixo risco
  "card1": 7890,
  "card2": 789,
  "card3": 789,
  "card4": 789,
  "card5": 789,
  "card6": "credit",
  "addr1": 789,
  "addr2": 789,
  "dist1": 8, // Distância pequena
  "dist2": 8, // Consistente
  "P_emaildomain": "yahoo.com",
  "R_emaildomain": "yahoo.com", // Mesmo domínio
  "M1": "T",
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

const nonFraudCase8 = {
  "TransactionID": "legit-test-008",
  "TransactionDT": 61200, // 17 PM - fim de expediente
  "TransactionAmt": 35.00, // Valor baixo 
  "ProductCD": "C", // Baixo risco
  "card1": 9012,
  "card2": 901,
  "card3": 901,
  "card4": 901,
  "card5": 901,
  "card6": "debit",
  "addr1": 901,
  "addr2": 901,
  "dist1": 5, // Distância pequena (reduzida)
  "dist2": 5, // Consistente (reduzida)
  "P_emaildomain": "icloud.com",
  "R_emaildomain": "icloud.com", // Mesmo domínio
  "M1": "T",
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

const nonFraudCase9 = {
  "TransactionID": "legit-test-009",
  "TransactionDT": 64800, // 18 PM - fim de dia
  "TransactionAmt": 15.99, // Valor muito baixo 
  "ProductCD": "C", // Baixo risco
  "card1": 1357,
  "card2": 135,
  "card3": 135,
  "card4": 135,
  "card5": 135,
  "card6": "credit",
  "addr1": 135,
  "addr2": 135,
  "dist1": 3, // Distância muito pequena (reduzida)
  "dist2": 3, // Consistente (reduzida)
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com", // Mesmo domínio
  "M1": "T",
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

const nonFraudCase10 = {
  "TransactionID": "legit-test-010",
  "TransactionDT": 39600, // 11 AM - horário normal
  "TransactionAmt": 59.99, // Valor baixo 
  "ProductCD": "C", // Baixo risco 
  "card1": 2468,
  "card2": 246,
  "card3": 246,
  "card4": 246,
  "card5": 246,
  "card6": "debit",
  "addr1": 246,
  "addr2": 246,
  "dist1": 10, // Distância pequena 
  "dist2": 10, // Consistente 
  "P_emaildomain": "outlook.com",
  "R_emaildomain": "outlook.com", // Mesmo domínio
  "M1": "T",
  "M2": "T",
  "M3": "T",
  "M4": "M0",
  "M5": "F",
  "M6": "T",
  "M7": "F",
  "M8": "T",
  "M9": "F"
};

// Adicionar variáveis V para cada novo caso
// Para casos de fraude: valores negativos nas importantes
const allFraudCases = [fraudCase1, fraudCase2, fraudCase3, fraudCase4, fraudCase5, 
                      fraudCase6, fraudCase7, fraudCase8, fraudCase9, fraudCase10];

// Para casos não fraude: valores positivos nas importantes
const allNonFraudCases = [nonFraudCase1, nonFraudCase2, nonFraudCase3, nonFraudCase4, nonFraudCase5,
                         nonFraudCase6, nonFraudCase7, nonFraudCase8, nonFraudCase9, nonFraudCase10];

// Adicionar variáveis V para os casos de fraude
for (const fraudCase of allFraudCases) {
  for (let i = 1; i <= 339; i++) {
    if (i <= 20) {
      fraudCase[`V${i}`] = Math.random() * -2 - 0.5; // -0.5 a -2.5
    } else {
      fraudCase[`V${i}`] = Math.random() * 2 - 1; // -1 a 1
    }
  }
}

// Adicionar variáveis V para os casos de não fraude
for (const nonFraudCase of allNonFraudCases) {
  for (let i = 1; i <= 339; i++) {
    if (i <= 20) {
      nonFraudCase[`V${i}`] = Math.random() * 2 + 0.5; // 0.5 a 2.5
    } else {
      nonFraudCase[`V${i}`] = Math.random() * 2 - 1; // -1 a 1
    }
  }
}

// Função para testar um caso específico e coletar resultados para cálculo de precisão
async function testCase(transaction, description, expectedResult) {
  console.log('=================================================================');
  console.log(`TESTE: ${description}`);
  console.log(`TransactionID: ${transaction.TransactionID}`);
  console.log(`Resultado Esperado: ${expectedResult}`);
  console.log('=================================================================');
  
  try {
    const response = await fetch('http://localhost:3000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(transaction)
    });
    
    const responseText = await response.text();
    
    try {
      // Tentar analisar a resposta como JSON
      const data = JSON.parse(responseText);
      
      let testResult = { expected: expectedResult, actual: null, correct: false };
      
      if (data.prediction) {
        const isFraud = data.prediction.is_fraud;
        const actualResult = isFraud ? 'FRAUDE' : 'NÃO-FRAUDE';
        testResult.actual = actualResult;
        testResult.correct = (expectedResult === actualResult);
        testResult.probability = data.prediction.fraud_probability;
        
        console.log('RESULTADO:');
        console.log(`É fraude? ${isFraud ? 'SIM' : 'NÃO'}`);
        console.log(`Probabilidade: ${(data.prediction.fraud_probability * 100).toFixed(2)}%`);
        console.log(`Threshold usado: ${data.prediction.threshold}`);
        console.log(`Corresponde ao esperado? ${testResult.correct ? 'SIM ✓' : 'NÃO ✗'}`);
      } else if (data.error) {
        console.log('ERRO:', data.error);
        if (data.traceback) {
          console.log('DETALHES DO ERRO:');
          console.log(data.traceback);
        }
        testResult.actual = 'ERRO';
        testResult.error = data.error;
      }
      
      console.log('\n');
      return testResult;
    } catch (parseError) {
      console.error('Erro ao processar resposta do servidor:');
      console.error(`Resposta bruta: ${responseText}`);
      console.error(`Erro de parse: ${parseError.message}`);
      return { expected: expectedResult, actual: 'ERRO', correct: false, error: parseError.message };
    }
  } catch (error) {
    console.error('Erro ao testar caso:', error.message);
    return { expected: expectedResult, actual: 'ERRO', correct: false, error: error.message };
  }
}

// Função principal de teste
async function runTests() {
  try {
    // Verificar se a API está online
    console.log('Verificando se a API está disponível...');
    const healthResponse = await fetch('http://localhost:3000/api/health');
    const healthData = await healthResponse.json();
    console.log('Status da API:', healthData.status);
    
    if (healthData.status !== 'OK') {
      console.error('API não está pronta. Verifique se o servidor está rodando.');
      return;
    }
    
    console.log('\nIniciando testes de casos...\n');
    
    const testResults = [];
    
    // Testar casos de fraude
    for (let i = 0; i < allFraudCases.length; i++) {
      const fraudCase = allFraudCases[i];
      const result = await testCase(fraudCase, `Caso de Fraude #${i+1}`, "FRAUDE");
      testResults.push(result);
    }
    
    // Testar casos de não-fraude
    for (let i = 0; i < allNonFraudCases.length; i++) {
      const nonFraudCase = allNonFraudCases[i];
      const result = await testCase(nonFraudCase, `Caso de Não-Fraude #${i+1}`, "NÃO-FRAUDE");
      testResults.push(result);
    }
    
    // Calcular e exibir as métricas de desempenho
    const totalTests = testResults.length;
    const correctTests = testResults.filter(r => r.correct).length;
    const accuracy = (correctTests / totalTests) * 100;
    
    // Contagens específicas para fraude e não-fraude
    const fraudTests = testResults.filter(r => r.expected === "FRAUDE");
    const nonFraudTests = testResults.filter(r => r.expected === "NÃO-FRAUDE");
    
    const truePositives = fraudTests.filter(r => r.actual === "FRAUDE").length;
    const falseNegatives = fraudTests.filter(r => r.actual === "NÃO-FRAUDE").length;
    
    const trueNegatives = nonFraudTests.filter(r => r.actual === "NÃO-FRAUDE").length;
    const falsePositives = nonFraudTests.filter(r => r.actual === "FRAUDE").length;
    
    // Métricas adicionais
    const fraudRecall = truePositives / fraudTests.length * 100;
    const fraudPrecision = truePositives / (truePositives + falsePositives) * 100 || 0;
    const nonFraudRecall = trueNegatives / nonFraudTests.length * 100;
    
    // Matriz de confusão
    console.log('\n==================== RESULTADOS ====================');
    console.log(`Total de testes realizados: ${totalTests}`);
    console.log(`Testes corretos: ${correctTests}`);
    console.log(`Precisão geral: ${accuracy.toFixed(2)}%`);
    console.log('\n------------------ MATRIZ DE CONFUSÃO ------------------');
    console.log('                | Previsto FRAUDE | Previsto NÃO-FRAUDE');
    console.log(`Real FRAUDE     |       ${truePositives.toString().padStart(2)}        |        ${falseNegatives.toString().padStart(2)}`);
    console.log(`Real NÃO-FRAUDE |       ${falsePositives.toString().padStart(2)}        |        ${trueNegatives.toString().padStart(2)}`);
    console.log('\n------------------ MÉTRICAS DETALHADAS ------------------');
    console.log(`Recall para casos de fraude: ${fraudRecall.toFixed(2)}%`);
    console.log(`Precisão para casos de fraude: ${fraudPrecision.toFixed(2)}%`);
    console.log(`Recall para casos não-fraude: ${nonFraudRecall.toFixed(2)}%`);
    console.log('=========================================================');
    
    console.log('\nTodos os testes foram concluídos!');
    
  } catch (error) {
    console.error('Erro ao executar testes:', error);
  }
}

// Executar os testes
console.log('Iniciando suíte de testes completa...');
runTests(); 