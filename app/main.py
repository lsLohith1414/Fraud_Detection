import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# =====================================================
# 1️⃣ Configure MLflow (DagsHub)
# =====================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TRACKING_URI:
    raise ValueError("MLFLOW_TRACKING_URI not set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "network_security_lgbm"
MODEL_URI = f"models:/{MODEL_NAME}@production"


# =====================================================
# 2️⃣ Load Production Model at Startup
# =====================================================
print("🔄 Loading production model from MLflow...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("✅ Model loaded successfully.")


# =====================================================
# 3️⃣ Initialize FastAPI App
# =====================================================
app = FastAPI(
    title="Fraud Detection API",
    version="1.0"
)


# =====================================================
# 4️⃣ Input Schema (RAW FEATURES ONLY)
# =====================================================
class FraudPredictionInput(BaseModel):
    TransactionID: int
    Amount: float
    CustomerID: int
    Timestamp: str
    MerchantID: int
    TransactionAmount: float
    AnomalyScore: float
    Category: str
    MerchantName: str
    Location: str
    Name: str
    Age: int
    Address: str
    AccountBalance: float
    LastLogin: str
    SuspiciousFlag: int


# =====================================================
# 5️⃣ Health Check
# =====================================================
@app.get("/")
def health():
    return {"status": "API is running successfully"}


# =====================================================
# 6️⃣ Prediction Endpoint
# =====================================================
@app.post("/predict")
def predict(data: FraudPredictionInput):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict
    prediction = model.predict(input_df)

    # Probability (binary classifier)
    try:
        probability = model.predict_proba(input_df)[:, 1]
        fraud_probability = float(probability[0])
    except Exception:
        fraud_probability = None

    return {
        "prediction": int(prediction[0]),
        "fraud_probability": fraud_probability
    }
