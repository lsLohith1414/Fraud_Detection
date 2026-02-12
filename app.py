from prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()


import pandas as pd

new_raw_data = pd.DataFrame([{
    "TransactionID": 1,
    "Amount": 55.530334,
    "CustomerID": 1952,
    "FraudIndicator": 0,        # will NOT be used in prediction
    "Timestamp": "2022-01-01 00:00:00",
    "MerchantID": 2701,
    "TransactionAmount": 79.413607,
    "AnomalyScore": 0.686699,
    "Category": "Other",
    "MerchantName": "Merchant 2701",
    "Location": "Location 2701",
    "Name": "Customer 1952",
    "Age": 50,
    "Address": "Address 1952",
    "AccountBalance": 2869.689912,
    "LastLogin": "2024-08-09",
    "SuspiciousFlag": 0         # will NOT be used in prediction
}])


pred, prob = pipeline.predict(new_raw_data)

print("Fraud Prediction:", pred)
print("Fraud Probability:", prob)
