import os
import json
import yaml
import mlflow
import pandas as pd

from imblearn.pipeline import Pipeline  # ✅ MUST be imblearn
from imblearn.combine import SMOTEENN

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)

from src.common.logger import get_logger
from src.common.exception import CustomException
from src.entities.config.model_trainer_config import ModelTrainerConfig
from src.entities.artifact.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.inference.feature_engineering import FeatureEngineeringTransformer

logger = get_logger(__name__)


# ======================================================
# MODEL TRAINER
# ======================================================
class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.config = model_trainer_config
        self.data_artifact = data_transformation_artifact

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not tracking_uri or not username or not password:
            raise ValueError("MLFLOW credentials not set properly")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Network_security_main")

        self.client = MlflowClient()

    # --------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Model training started")

            # ================= LOAD CLEANED RAW DATA =================
            train_df = pd.read_csv(self.data_artifact.transformed_train_file_path)
            test_df = pd.read_csv(self.data_artifact.transformed_test_file_path)

            X_train = train_df.drop(columns=[self.config.target_column])
            y_train = train_df[self.config.target_column]

            X_test = test_df.drop(columns=[self.config.target_column])
            y_test = test_df[self.config.target_column]

            params = self.config.params
            model_name = self.config.model_name

            # ======================================================
            # PREPROCESSING PIPELINES
            # ======================================================

            robust_features = [
                "Amount",
                "TransactionAmount",
                "AccountBalance",
                "user_avg_amount",
                "amount_last_24h",
                "merchant_avg_amount",
                "amount_diff_from_avg",
                "amount_diff_from_merchant_avg",
                "user_txn_count",
                "user_daily_txns",
                "txns_last_1h",
                "txns_last_24h",
                "merchant_txn_count",
                "days_since_last_login",
                "txn_gap_minutes",
            ]

            minmax_features = [
                "AnomalyScore",
                "amount_ratio_avg",
                "Age",
            ]

            categorical_features = ["Category"]

            robust_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ])

            minmax_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])

            preprocessor = ColumnTransformer([
                ("robust", robust_pipeline, robust_features),
                ("minmax", minmax_pipeline, minmax_features),
                ("cat", categorical_pipeline, categorical_features),
            ])

            # ======================================================
            # FULL TRAINING PIPELINE
            # ======================================================

            full_pipeline = Pipeline([
                ("feature_engineering", FeatureEngineeringTransformer()),
                ("preprocessing", preprocessor),
                ("smote", SMOTEENN(random_state=42)),
                ("model", LGBMClassifier(**params)),
            ])

            logger.info("Training full pipeline...")

            with mlflow.start_run(run_name=model_name):

                # 🔥 TRAIN
                full_pipeline.fit(X_train, y_train)

                # 🔥 EVALUATE
                y_pred = full_pipeline.predict(X_test)
                y_proba = full_pipeline.predict_proba(X_test)[:, 1]

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, zero_division=0),
                    "pr_auc": average_precision_score(y_test, y_proba),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "classification_report": classification_report(
                        y_test, y_pred, zero_division=0
                    ),
                }

                # ================= LOGGING =================

                mlflow.log_params(params)
                mlflow.log_metrics({
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "pr_auc": metrics["pr_auc"],
                })

                mlflow.log_text(
                    json.dumps(metrics["confusion_matrix"].tolist()),
                    "confusion_matrix.json",
                )

                mlflow.log_text(
                    metrics["classification_report"],
                    "classification_report.txt",
                )

                # 🔥 Log entire pipeline
                mlflow.sklearn.log_model(
                    full_pipeline,
                    artifact_path="model",
                    registered_model_name="network_security_lgbm",
                )

                # ================= SAVE LOCALLY =================
                artifact_dir = os.path.dirname(self.config.trained_model_path)
                os.makedirs(artifact_dir, exist_ok=True)

                # Convert numpy types before saving
                safe_metrics = {
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1_score": float(metrics["f1_score"]),
                    "pr_auc": float(metrics["pr_auc"]),
                    "confusion_matrix": metrics["confusion_matrix"].tolist(),
                    "classification_report": metrics["classification_report"],
                }

                with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
                    json.dump(safe_metrics, f, indent=4)

                with open(os.path.join(artifact_dir, "metrics.yaml"), "w") as f:
                    yaml.dump(safe_metrics, f)

            logger.info("Model training completed successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_path,
                train_metric_artifact=None,
                test_metric_artifact=ClassificationMetricArtifact(
                    precision_score=metrics["precision"],
                    recall_score=metrics["recall"],
                    f1_score=metrics["f1_score"],
                ),
            )

        except Exception as e:
            logger.error("Model training failed", exc_info=True)
            raise CustomException(e)


# ======================================================
# MAIN
# ======================================================
def main():
    try:
        from src.common.utils import read_yaml
        from src.entities.config.training_pipeline_config import (
            TrainingPipelineConfig,
        )
        from src.entities.config.data_transformation_config import (
            DataTransformationConfig,
        )

        config = read_yaml(os.path.join("config", "config.yaml"))

        training_pipeline_config = TrainingPipelineConfig(config=config)
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config
        )

        data_transformation_artifact = DataTransformationArtifact(
            transformed_object_file_path=None,  # ❌ no preprocessor file now
            transformed_train_file_path=data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=data_transformation_config.transformed_test_file_path,
        )

        model_trainer_config = ModelTrainerConfig(
            training_pipeline_config=training_pipeline_config
        )

        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )

        model_trainer.initiate_model_trainer()

    except Exception as e:
        logger.error("Training pipeline failed", exc_info=True)
        raise CustomException(e)


if __name__ == "__main__":
    main()
