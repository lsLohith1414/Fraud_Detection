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
# SAFE PARAM PARSER (FIXES ALL THESE ERRORS)
# ======================================================
def parse_mlflow_params(raw_params: dict) -> dict:
    """
    Convert MLflow string params to correct Python types
    and DROP invalid None-like parameters.
    """
    parsed = {}

    for k, v in raw_params.items():
        if v is None:
            continue

        if isinstance(v, str):
            v_strip = v.strip().lower()

            # DROP explicit "none"
            if v_strip == "none":
                continue

            if v_strip == "true":
                parsed[k] = True
            elif v_strip == "false":
                parsed[k] = False
            else:
                # Try numeric
                try:
                    if "." in v_strip:
                        parsed[k] = float(v_strip)
                    else:
                        parsed[k] = int(v_strip)
                except ValueError:
                    parsed[k] = v  # keep string

        else:
            parsed[k] = v

    return parsed


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
    # FETCH APPROVED PARAMS
    # --------------------------------------------------
    def _get_approved_params(self):
        model_version = self.client.get_model_version_by_alias(
            name=self.config.model_name,
            alias=self.config.model_alias
        )

        run = self.client.get_run(model_version.run_id)

        raw_params = run.data.params
        params = parse_mlflow_params(raw_params)

        logger.info(
            f"Using model '{self.config.model_name}' "
            f"version {model_version.version} "
            f"(alias={self.config.model_alias})"
        )

        logger.info(f"Final model params: {params}")

        return params, model_version

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
