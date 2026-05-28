from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# ── LOAD MODEL ────────────────────────────────────────────────
model = pickle.load(open(r'C:\Users\vicen\Desktop\fraud-detection\model.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\vicen\Desktop\fraud-detection\scaler.pkl', 'rb'))

app = FastAPI(title="Credit Card Fraud Detector API")

# ── INPUT SCHEMA ──────────────────────────────────────────────
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

# ── ENDPOINTS ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detector API", "status": "running"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array([[
        transaction.Time,
        transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
        transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
        transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
        transaction.V26, transaction.V27, transaction.V28,
        transaction.Amount
    ]])
    
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0]
    
    return {
        "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "confidence": round(float(probability[1] if prediction == 1 else probability[0]) * 100, 2),
        "fraud_probability": round(float(probability[1]) * 100, 2)
    }