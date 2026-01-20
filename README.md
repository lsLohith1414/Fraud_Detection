# 🚨 Fraud Detection System

A production-grade Machine Learning system designed to detect fraudulent financial transactions using end-to-end ML pipelines, advanced preprocessing, imbalanced learning techniques, model evaluation, and deployable FastAPI service. This repository follows real-world industry standards including modular code structure, MLflow experiment tracking, artifact storage, and scalable deployment design.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [ML Pipeline](#ml-pipeline)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [API Deployment](#api-deployment)
- [Folder Structure](#folder-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🧾 Overview

Fraudulent financial transactions are rare, unpredictable, and often lack clear patterns. This project implements a **production-ready fraud detection system** that includes:

✔ End-to-end ML pipeline  
✔ Imbalanced data handling (SMOTE, ENN, SMOTEENN)  
✔ Feature preprocessing & transformation  
✔ Outlier detection  
✔ MLflow experiment tracking  
✔ Deployable prediction API using FastAPI  

This project aims to build a scalable, reliable, and explainable fraud detection solution suitable for real-world deployment.

---

## ✨ Features

- Full ML pipeline (preprocessing → training → evaluation → deployment)  
- Automated logging using **MLflow**  
- Advanced imbalanced data handling  
- Feature scaling, encoding, and transformation  
- Outlier detection using IQR / Isolation Forest  
- Model explainability using SHAP  
- REST API using FastAPI for real-time predictions  
- Clean & modular industry-standard code structure  
- Docker-ready (optional)  
- DVC-ready (optional)

---

## 🛠 Tech Stack

- **Python 3.10+**  
- Scikit-learn  
- XGBoost / LightGBM  
- Pandas & NumPy  
- MLflow  
- FastAPI + Uvicorn  
- Matplotlib / Seaborn  
- DVC (optional)

---

## 🏗 Project Architecture

Raw Data → Preprocessing → Resampling → Feature Engineering  
→ Model Training → Evaluation → Model Registry (MLflow)  
→ FastAPI Deployment

```yaml
# Example architecture visualization (placeholder)
Raw Data:
    - raw/*.csv

Preprocessing:
    - missing value handling
    - encoding
    - scaling
    - transformations

Resampling:
    - SMOTE / ENN / SMOTEENN

Model Training:
    - ML algorithms
    - hyperparameter tuning

Evaluation:
    - PR-AUC
    - ROC-AUC
    - Precision/Recall

Model Registry:
    - MLflow tracking

Deployment:
    - FastAPI
    - Docker (optional)
```

---

## 📊 Dataset Description

The dataset contains key transaction features such as:

| Feature        | Description                                |
|----------------|--------------------------------------------|
| TransactionID  | Unique transaction ID                      |
| Amount         | Transaction value                          |
| CustomerID     | Customer identifier                        |
| MerchantID     | Merchant identifier                        |
| Timestamp      | Date & time of transaction                 |
| Category       | Transaction type                           |
| AnomalyScore   | Engine-generated anomaly score             |
| FraudIndicator | Target variable (0 = Legit, 1 = Fraud)     |

> Most fraud datasets are **highly imbalanced**, so hybrid resampling is crucial.

---

## ⚙ Installation

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

---

## ▶ How to Run

### 1️⃣ Run preprocessing
```bash
python src/pipeline/preprocess.py
```

### 2️⃣ Train model
```bash
python src/pipeline/train.py
```

### 3️⃣ Evaluate model
```bash
python src/pipeline/evaluate.py
```

### 4️⃣ Start API
```bash
uvicorn src.api.app:app --reload
```

---

## 🔄 ML Pipeline

### 1. Missing Value Imputation
- Numerical: Mean/Median  
- Categorical: Most Frequent  

### 2. Scaling
- StandardScaler  
- MinMaxScaler  

### 3. Transformation
- Log transform for skewed features  
- Quantile Transformer (optional)

### 4. Outlier Detection
- IQR method  
- Isolation Forest  

### 5. Resampling
- SMOTE  
- ENN  
- SMOTEENN  

### 6. Feature Selection
- Mutual Information  
- ANOVA F-test  
- Boruta (optional)

---

## 🧠 Model Training

Training uses several ML models:

- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  

MLflow logs:

- Hyperparameters  
- Metrics  
- Artifacts  
- Trained models  

---

## 📈 Evaluation

Important fraud detection metrics:

- **PR-AUC (primary metric)**  
- ROC-AUC  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- Calibration Curve  

Fraud detection focuses heavily on **recall** and **precision–recall tradeoff**.

---

## 🚀 API Deployment (FastAPI)

### Example Request:
```json
{
  "Amount": 850,
  "CustomerID": 1234,
  "MerchantID": 99,
  "Timestamp": "2025-01-20 14:22:00",
  "Category": "Online"
}
```

### Example Response:
```json
{
  "fraud_probability": 0.91,
  "is_fraud": 1
}
```

---

## 🗂 Folder Structure

```css
fraud-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── pipeline/
│   ├── utils/
│   └── api/
│
├── notebooks/
├── mlruns/
├── requirements.txt
└── README.md
```

---

## 🚧 Future Enhancements

- Real-time streaming (Kafka)  
- Ensemble stacking models  
- AutoML integration  
- Docker + Kubernetes deployment  
- AWS Lambda / Azure / GCP deployment  
- Advanced anomaly detection algorithms  

---

## 🤝 Contributing

Pull requests are welcome.  
Follow standard linting & coding guidelines before submitting.

---

## 📄 License

This project is licensed under the **MIT License**.

```yaml
# License placeholder
MIT License
```

---

