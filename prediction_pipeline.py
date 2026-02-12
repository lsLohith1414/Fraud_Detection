import joblib
import pandas as pd
import os 
from src.components.data_transformation import DataTransformation
from src.entities.config.training_pipeline_config import TrainingPipelineConfig 
from src.entities.config.data_validation_config import DataValidationConfig 

from src.entities.artifact.artifacts_entity import  DataValidationArtifact, DataTransformationArtifact

from src.entities.config.data_transformation_config import DataTransformationConfig
# from src.entities.artifact.artifacts_entity import DataIngestionAftifacts  
from src.common.utils import read_yaml
config = read_yaml(os.path.join("config", "config.yaml"))
training_pipeline_config = TrainingPipelineConfig(config=config)



data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
data_valid_train_path = data_validation_config.valid_train_file_path
data_valid_test_path = data_validation_config.valid_test_file_path
data_valid_drift_report = data_validation_config.drift_report_path

data_validation_artifact = DataValidationArtifact(valid_train_file_path=data_valid_train_path,valid_test_file_path=data_valid_test_path,drift_report_file_path=data_valid_drift_report)
data_transformation_config = DataTransformationConfig(training_pipeline_config)







class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load("model_preprocessor/model.pkl")
        self.preprocessor = joblib.load("model_preprocessor/preprocessor.pkl")
        self.fe = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
        

        self.final_features = [
            'Amount', 'TransactionAmount', 'AnomalyScore', 'Category', 'Age',
            'AccountBalance', 'txn_hour', 'txn_weekday', 'txn_month',
            'is_weekend', 'days_since_last_login', 'txn_gap_minutes',
            'user_avg_amount', 'amount_diff_from_avg', 'amount_ratio_avg',
            'user_txn_count', 'user_daily_txns', 'txns_last_1h',
            'txns_last_24h', 'amount_last_24h', 'merchant_txn_count',
            'merchant_avg_amount', 'amount_diff_from_merchant_avg',
            'is_new_location', 'is_high_risk_category'
        ]

        

        # print(len(self.final_features))

    def predict(self, raw_df: pd.DataFrame):
        # 1️⃣ Feature engineering
        df = self.fe.create_new_features_from_existing_features(df=raw_df)
        # print(df)

        # 2️⃣ Select same features used in training
        df = df[self.final_features]
        # print(df)

        # 3️⃣ Preprocessing (scaling, encoding, etc.)
        X = self.preprocessor.transform(df)
        print(len(X[0]))        

        # 4️⃣ Prediction
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]

        return preds, probs
