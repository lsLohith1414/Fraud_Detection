# import os 
# import pandas as pd

# from src.common.logger import get_logger
# logger = get_logger(__name__)
# from src.common.exception import CustomException


# from src.entities.config.data_transformation_config import DataTransformationConfig

# path_train_df_path = os.path.join("artifacts","01_28_2026_14_15","data_validation","valid","train.csv")

# train_df = pd.read_csv(path_train_df_path)

# print(train_df)


# def create_new_features_from_existing(df:pd.DataFrame) -> pd.DataFrame:
#     pass


     
# def create_new_features_from_existing_features(df:pd.DataFrame) -> pd.DataFrame:
#     # 1. Time based featues:

#     df["txn_hour"]     = df["Timestamp"].dt.hour
#     df["txn_weekday"]  = df["Timestamp"].dt.weekday   # 0=Mon
#     df["txn_month"]    = df["Timestamp"].dt.month
#     df["is_weekend"]   = df["txn_weekday"].isin([5,6]).astype(int)


#     ## 2. USER-LEVEL BEHAVIOR FEATURES

#     df['user_avg_amount'] = df.groupby('CustomerID')['TransactionAmount'].transform('mean')
#     df['amount_diff_from_avg'] = df['TransactionAmount'] - df['user_avg_amount']
#     df['amount_ratio_avg'] = df['TransactionAmount'] / (df['user_avg_amount'] + 1)
#     df['user_txn_count'] = df.groupby('CustomerID').cumcount() + 1
#     df['date'] = pd.to_datetime(df['Timestamp']).dt.date
#     df['user_daily_txns'] = df.groupby(['CustomerID', 'date']).cumcount() + 1



#         ## 3. VELOCITY FEATURES
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#     df = df.sort_values(['CustomerID', 'Timestamp'])




#     # Transactions in the Past 1 Hour
#     # Convert timestamp
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#     df = df.dropna(subset=['Timestamp'])
    
#     # Sort properly
#     df = df.sort_values(['CustomerID', 'Timestamp']).reset_index(drop=True)
    
#     # Empty column
#     df['txns_last_1h'] = 0
    
#     # Loop per customer (FAST because each customer has few transactions)
#     for cid, group in df.groupby('CustomerID'):
#         times = group['Timestamp'].values
#         result = np.zeros(len(times), dtype=int)
        
#         for i in range(len(times)):
#             # find how many timestamps fall within [current_time - 1h, current_time]
#             cutoff = times[i] - np.timedelta64(1, 'h')
#             result[i] = np.sum((times >= cutoff) & (times <= times[i]))
        
#         df.loc[group.index, 'txns_last_1h'] = result



#     # Transactions in the Past 24 Hours
#     # Make sure timestamps are clean
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#     df = df.dropna(subset=['Timestamp'])
    
#     # Sort properly
#     df = df.sort_values(['CustomerID', 'Timestamp']).reset_index(drop=True)

#     # Create empty columns
#     df['txns_last_24h'] = 0
#     df['amount_last_24h'] = 0.0
    
#     # Loop per customer (FAST because each customer has few transactions)
#     for cid, group in df.groupby('CustomerID'):
#         times = group['Timestamp'].values
#         amounts = group['TransactionAmount'].values
    
#         txn_count = np.zeros(len(times), dtype=int)
#         amount_sum = np.zeros(len(times), dtype=float)
    
#         for i in range(len(times)):
#             cutoff = times[i] - np.timedelta64(24, 'h')
    
#             mask = (times >= cutoff) & (times <= times[i])
    
#             txn_count[i] = np.sum(mask)
#             amount_sum[i] = np.sum(amounts[mask])
    
#         df.loc[group.index, 'txns_last_24h'] = txn_count
#         df.loc[group.index, 'amount_last_24h'] = amount_sum



#     # Amount spent in the last 24 hours
#     # Ensure timestamp is cleaned and sorted
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#     df = df.dropna(subset=['Timestamp'])
    
#     df = df.sort_values(['CustomerID', 'Timestamp']).reset_index(drop=True)
    
#     # Create empty output column
#     df['amount_last_24h'] = 0.0
    
#     # Loop per customer ID
#     for cid, group in df.groupby('CustomerID'):
#         times = group['Timestamp'].values
#         amounts = group['TransactionAmount'].values
    
#         amount_sum_24h = np.zeros(len(times), dtype=float)
    
#         for i in range(len(times)):
#             cutoff = times[i] - np.timedelta64(24, 'h')
    
#             # Mask of transactions within [current_time - 24h, current_time]
#             mask = (times >= cutoff) & (times <= times[i])
    
#             amount_sum_24h[i] = np.sum(amounts[mask])
    
#         # Assign back to main df
#         df.loc[group.index, 'amount_last_24h'] = amount_sum_24h



#     ## 4. MERCHANT-LEVEL RISK FEATURES
    
#     df['merchant_txn_count'] = df.groupby('MerchantID')['TransactionAmount'].transform('count')
#     df['merchant_avg_amount'] = df.groupby('MerchantID')['TransactionAmount'].transform('mean')
#     df['amount_diff_from_merchant_avg'] = df['TransactionAmount'] - df['merchant_avg_amount']
    
    
    
    
    
#     # 5. LOCATION & DEVICE FEATURES
    
#     df['is_new_location'] = (
#         df.groupby('CustomerID')['Location']
#               .transform(lambda x: ~x.duplicated().astype(int))
#     )
    
    
#     # 6. CATEGORY RISK FEATURES
    
#     high_risk = ['electronics','jewelry','crypto']
#     df['is_high_risk_category'] = df['Category'].isin(high_risk).astype(int)
















import os
import dagshub
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score
)

from src.common.logger import get_logger
from src.common.exception import CustomException
from src.entities.config.model_trainer_config import ModelTrainerConfig
from src.entities.artifact.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)

logger = get_logger(__name__)


# ======================================================
# MODEL EVALUATION
# ======================================================
def evaluate_classification_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }


# ======================================================
# MODEL TRAINER
# ======================================================
class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        self.config = model_trainer_config
        self.data_artifact = data_transformation_artifact

        dagshub.init( 
            repo_owner="lsLohith1414",
            repo_name="Fraud_Detection",
            mlflow=True
        )

        self.client = MlflowClient()


    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Model training pipeline started")

            # ================= LOAD DATA =================
            train_df = pd.read_csv(self.data_artifact.transformed_train_file_path)
            test_df = pd.read_csv(self.data_artifact.transformed_test_file_path)

            X_train = train_df.drop(columns=[self.config.target_column])
            y_train = train_df[self.config.target_column]

            X_test = test_df.drop(columns=[self.config.target_column])
            y_test = test_df[self.config.target_column]

            # ================= FETCH PARAMS =================
            params, model_version = self._get_approved_params()

            # ================= BUILD MODEL =================
            model = LGBMClassifier(**params)

            # ================= TRAIN + EVAL =================
            with mlflow.start_run(run_name="production_training"):

                mlflow.set_tag("pipeline_stage", "model_training")
                mlflow.set_tag("model_name", self.config.model_name)
                mlflow.set_tag("model_alias", self.config.model_alias)
                mlflow.set_tag("source_run_id", model_version.run_id)

                logger.info("Training model")
                model.fit(X_train, y_train)

                logger.info("Evaluating model")
                eval_results = evaluate_classification_model(model, X_test, y_test)

                for k, v in eval_results.items():
                    if isinstance(v, (float, int)):
                        mlflow.log_metric(k, v)

                logger.info(f"PR-AUC    : {eval_results['pr_auc']}")
                logger.info(f"F1 Score  : {eval_results['f1_score']}")
                logger.info(f"Recall    : {eval_results['recall']}")
                logger.info(f"Precision : {eval_results['precision']}")

                logger.info("Confusion Matrix:\n" + str(eval_results["confusion_matrix"]))
                logger.info(
                    "Classification Report:\n"
                    + eval_results["classification_report"]
                )

                # ================= SAVE MODEL =================
                os.makedirs(
                    os.path.dirname(self.config.trained_model_path),
                    exist_ok=True
                )

                mlflow.sklearn.save_model(
                    sk_model=model,
                    path=self.config.trained_model_path
                )

            logger.info("Model training pipeline completed successfully")

            test_metrics = ClassificationMetricArtifact(
                precision_score=eval_results["precision"],
                recall_score=eval_results["recall"],
                f1_score=eval_results["f1_score"]
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_path,
                train_metric_artifact=None,
                test_metric_artifact=test_metrics
            )

        except Exception as e:
            logger.error("Model training failed", exc_info=True)
            raise CustomException(e)









def main():

    try: 

        from src.common.utils import read_yaml
        from src.entities.config.training_pipeline_config import TrainingPipelineConfig
        from src.entities.config.data_transformation_config import DataTransformationConfig
        from src.entities.artifact.artifacts_entity import DataTransformationArtifact


        config = read_yaml(os.path.join("config", "config.yaml"))

        training_pipeline_config = TrainingPipelineConfig(config=config)

        data_transformation_config = DataTransformationConfig(training_pipeline_config)



        data_transformation_artifact = DataTransformationArtifact(transformed_object_file_path=data_transformation_config.preprocessor_file_path ,
                                                                  transformed_train_file_path=data_transformation_config.transformed_train_file_path,
                                                                  transformed_test_file_path=data_transformation_config.transformed_test_file_path)



        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)


        model_trainer = ModelTrainer(model_trainer_config= model_trainer_config,data_transformation_artifact=data_transformation_artifact)

        model_trainer.initiate_model_trainer()



    except Exception as e:
        logger.error("Data ingestion stage failed", exc_info=True)
        raise CustomException(e)



if __name__ == "__main__":
    main()