import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# 1️⃣ Environment Configuration
# -----------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

MODEL_NAME = "fraud_detection_model"
MODEL_URI = f"models:/{MODEL_NAME}@production"


# -----------------------------
# 2️⃣ Load Model at Startup
# -----------------------------
print("Loading production model from MLflow...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model loaded successfully.")


# -----------------------------
# 3️⃣ Define FastAPI App
# -----------------------------
app = FastAPI(title="Fraud Detection API")


# -----------------------------
# 4️⃣ Define Input Schema
# -----------------------------
class PredictionInput(BaseModel):
    TransactionID: int
    Amount: float
    CustomerID: int
    MerchantID: int
    TransactionAmount: float
    AnomalyScore: float


# -----------------------------
# 5️⃣ Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PredictionInput):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)

    return {
        "prediction": int(prediction[0])
    }
