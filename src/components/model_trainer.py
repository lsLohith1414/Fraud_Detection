# import os
# import json
# import yaml
# import mlflow
# import pandas as pd

# from lightgbm import LGBMClassifier
# from mlflow.tracking import MlflowClient

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
#     confusion_matrix,
#     average_precision_score,
# )

# from src.common.logger import get_logger
# from src.common.exception import CustomException
# from src.entities.config.model_trainer_config import ModelTrainerConfig
# from src.entities.artifact.artifacts_entity import (
#     DataTransformationArtifact,
#     ModelTrainerArtifact,
#     ClassificationMetricArtifact,
# )

# logger = get_logger(__name__)


# # ======================================================
# # MODEL EVALUATION
# # ======================================================
# def evaluate_classification_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]

#     return {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred),
#         "recall": recall_score(y_test, y_pred),
#         "f1_score": f1_score(y_test, y_pred),
#         "pr_auc": average_precision_score(y_test, y_proba),
#         "confusion_matrix": confusion_matrix(y_test, y_pred),
#         "classification_report": classification_report(y_test, y_pred),
#     }


# # ======================================================
# # MODEL TRAINER
# # ======================================================
# class ModelTrainer:
#     def __init__(
#         self,
#         model_trainer_config: ModelTrainerConfig,
#         data_transformation_artifact: DataTransformationArtifact,
#     ):
#         self.config = model_trainer_config
#         self.data_artifact = data_transformation_artifact

#         username = os.getenv("MLFLOW_TRACKING_USERNAME")
#         password = os.getenv("MLFLOW_TRACKING_PASSWORD")

#         if not username or not password:
#             raise ValueError(
#                 "MLFLOW_TRACKING_USERNAME or MLFLOW_TRACKING_PASSWORD not set"
#             )

#         mlflow.set_tracking_uri(
#             "https://dagshub.com/lsLohith1414/Fraud_Detection.mlflow"
#         )
#         mlflow.set_experiment("Network_security_main")

#         self.client = MlflowClient()
#         print("pass")

#     # --------------------------------------------------
#     def initiate_model_trainer(self) -> ModelTrainerArtifact:
#         try:
#             logger.info("Model training pipeline started")

#             # ================= LOAD DATA =================
#             train_df = pd.read_csv(self.data_artifact.transformed_train_file_path)
#             test_df = pd.read_csv(self.data_artifact.transformed_test_file_path)

#             X_train = train_df.drop(columns=[self.config.target_column])
#             y_train = train_df[self.config.target_column]

#             X_test = test_df.drop(columns=[self.config.target_column])
#             y_test = test_df[self.config.target_column]

#             params = self.config.params
#             model_name = self.config.model_name

#             logger.info(f"Training {model_name} with params {params}")

#             with mlflow.start_run(run_name=model_name):

#                 # ================= TRAIN =================
#                 model = LGBMClassifier(**params)
#                 model.fit(X_train, y_train)

#                 # ================= EVALUATE =================
#                 metrics = evaluate_classification_model(model, X_test, y_test)

#                 # ================= LOG TO MLFLOW =================
#                 mlflow.log_metrics(
#                     {
#                         "accuracy": metrics["accuracy"],
#                         "precision": metrics["precision"],
#                         "recall": metrics["recall"],
#                         "f1_score": metrics["f1_score"],
#                         "pr_auc": metrics["pr_auc"],
#                     }
#                 )

#                 mlflow.log_params(params)

#                 mlflow.log_text(
#                     json.dumps(metrics["confusion_matrix"].tolist()),
#                     "confusion_matrix.json",
#                 )

#                 mlflow.log_text(
#                     metrics["classification_report"], "classification_report.txt"
#                 )

#                 mlflow.sklearn.log_model(
#                     model,
#                     artifact_path="model",
#                     registered_model_name="network_security_lgbm",
#                 )

#                 # ================= SAVE LOCALLY =================
#                 artifact_dir = os.path.dirname(self.config.trained_model_path)
#                 os.makedirs(artifact_dir, exist_ok=True)

#                 # Save model locally
#                 mlflow.sklearn.save_model(
#                     sk_model=model, path=self.config.trained_model_path
#                 )

#                 # Save numeric metrics as JSON
#                 numeric_metrics = {
#                     "accuracy": metrics["accuracy"],
#                     "precision": metrics["precision"],
#                     "recall": metrics["recall"],
#                     "f1_score": metrics["f1_score"],
#                     "pr_auc": metrics["pr_auc"],
#                 }

#                 with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
#                     json.dump(numeric_metrics, f, indent=4)

#                 # Save confusion matrix
#                 with open(
#                     os.path.join(artifact_dir, "confusion_matrix.json"), "w"
#                 ) as f:
#                     json.dump(metrics["confusion_matrix"].tolist(), f, indent=4)

#                 # Save classification report
#                 with open(
#                     os.path.join(artifact_dir, "classification_report.txt"), "w"
#                 ) as f:
#                     f.write(metrics["classification_report"])

#                 # Optional: Save YAML version
#                 with open(os.path.join(artifact_dir, "metrics.yaml"), "w") as f:
#                     yaml.dump(numeric_metrics, f)

#             logger.info("Model training completed successfully")

#             test_metrics = ClassificationMetricArtifact(
#                 precision_score=metrics["precision"],
#                 recall_score=metrics["recall"],
#                 f1_score=metrics["f1_score"],
#             )

#             return ModelTrainerArtifact(
#                 trained_model_file_path=self.config.trained_model_path,
#                 train_metric_artifact=None,
#                 test_metric_artifact=test_metrics,
#             )

#         except Exception as e:
#             logger.error("Model training failed", exc_info=True)
#             raise CustomException(e)


# # ======================================================
# # MAIN
# # ======================================================
# def main():
#     try:
#         from src.common.utils import read_yaml
#         from src.entities.config.training_pipeline_config import TrainingPipelineConfig
#         from src.entities.config.data_transformation_config import (
#             DataTransformationConfig,
#         )

#         config = read_yaml(os.path.join("config", "config.yaml"))

#         training_pipeline_config = TrainingPipelineConfig(config=config)
#         data_transformation_config = DataTransformationConfig(training_pipeline_config)

#         data_transformation_artifact = DataTransformationArtifact(
#             transformed_object_file_path=data_transformation_config.preprocessor_file_path,
#             transformed_train_file_path=data_transformation_config.transformed_train_file_path,
#             transformed_test_file_path=data_transformation_config.transformed_test_file_path,
#         )

#         model_trainer_config = ModelTrainerConfig(
#             training_pipeline_config=training_pipeline_config
#         )

#         model_trainer = ModelTrainer(
#             model_trainer_config=model_trainer_config,
#             data_transformation_artifact=data_transformation_artifact,
#         )

#         model_trainer.initiate_model_trainer()

#     except Exception as e:
#         logger.error("Training pipeline failed", exc_info=True)
#         raise CustomException(e)


# if __name__ == "__main__":
#     main()




import os
import json
import yaml
import joblib
import mlflow
import pandas as pd

from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
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
# MODEL EVALUATION
# ======================================================
def evaluate_classification_model(model, X_test, y_test):
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
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.config = model_trainer_config
        self.data_artifact = data_transformation_artifact

        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if not username or not password or not tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI, USERNAME or PASSWORD not set"
            )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Network_security_main")

        self.client = MlflowClient()

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

            params = self.config.params
            model_name = self.config.model_name

            logger.info(f"Training {model_name} with params {params}")

            with mlflow.start_run(run_name=model_name):

                # ================= TRAIN MODEL =================
                model = LGBMClassifier(**params)
                model.fit(X_train, y_train)

                # ================= EVALUATE =================
                metrics = evaluate_classification_model(
                    model, X_test, y_test
                )

                # ================= LOAD PREPROCESSOR =================
                preprocessor = joblib.load(
                    self.data_artifact.transformed_object_file_path
                )

                # ================= BUILD FULL INFERENCE PIPELINE =================
                full_pipeline = Pipeline(steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    ("preprocessor", preprocessor),
                    ("model", model),
                ])

                # ================= LOG METRICS =================
                mlflow.log_metrics({
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "pr_auc": metrics["pr_auc"],
                })

                mlflow.log_params(params)

                mlflow.log_text(
                    json.dumps(metrics["confusion_matrix"].tolist()),
                    "confusion_matrix.json",
                )

                mlflow.log_text(
                    metrics["classification_report"],
                    "classification_report.txt",
                )

                # ================= LOG FULL PIPELINE =================
                mlflow.sklearn.log_model(
                    full_pipeline,
                    artifact_path="model",
                    registered_model_name="network_security_lgbm",
                )

                # ================= SAVE LOCALLY =================
                artifact_dir = os.path.dirname(
                    self.config.trained_model_path
                )
                os.makedirs(artifact_dir, exist_ok=True)

                joblib.dump(full_pipeline, self.config.trained_model_path)

                # Save numeric metrics locally
                numeric_metrics = {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "pr_auc": metrics["pr_auc"],
                }

                with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
                    json.dump(numeric_metrics, f, indent=4)

                with open(
                    os.path.join(artifact_dir, "confusion_matrix.json"), "w"
                ) as f:
                    json.dump(
                        metrics["confusion_matrix"].tolist(), f, indent=4
                    )

                with open(
                    os.path.join(artifact_dir, "classification_report.txt"),
                    "w",
                ) as f:
                    f.write(metrics["classification_report"])

                with open(os.path.join(artifact_dir, "metrics.yaml"), "w") as f:
                    yaml.dump(numeric_metrics, f)

            logger.info("Model training completed successfully")

            test_metrics = ClassificationMetricArtifact(
                precision_score=metrics["precision"],
                recall_score=metrics["recall"],
                f1_score=metrics["f1_score"],
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_path,
                train_metric_artifact=None,
                test_metric_artifact=test_metrics,
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
            transformed_object_file_path=data_transformation_config.preprocessor_file_path,
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
