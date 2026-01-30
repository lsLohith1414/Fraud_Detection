# src/components/data_validation.py

import os
import pandas as pd

from src.common.logger import get_logger
from src.common.exception import CustomException
from src.common.utils import (
    read_yaml,
    write_json,
    detect_data_drift
)
from src.entities.config.data_validation_config import DataValidationConfig
from src.entities.artifact.artifacts_entity import DataValidationArtifact
from src.entities.artifact.artifacts_entity import DataIngestionAftifacts

logger = get_logger(__name__)


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact:DataIngestionAftifacts):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.schema = read_yaml(self.data_validation_config.schema_path)


    def validate_schema(self, df: pd.DataFrame, dataset_name: str) -> bool:

        expected_columns = set(self.schema["features"])
        actual_columns = set(df.columns)

        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        if missing:
            logger.error(f"{dataset_name} missing columns: {missing}")
            return False

        if extra:
            logger.warning(f"{dataset_name} extra columns detected: {extra}")

        return True


    # =====================================================
    # STEP 2: Train / Test Structure Validation
    # =====================================================
    def validate_train_test_structure(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> bool:
        logger.info("Validating train-test schema consistency")

        if set(train_df.columns) != set(test_df.columns):
            logger.error("Train and Test datasets have different column structures")
            return False

        return True

    # =====================================================
    # STEP 3: Missing Value Check (Train / Test)
    # =====================================================
    def validate_missing_values(self, df: pd.DataFrame, dataset_name: str) -> bool:
        logger.info(f"Missing value check started for {dataset_name} dataset")

        threshold = self.schema["data_quality"]["missing_values"]["allowed_missing_ratio"]
        missing_ratio = df.isnull().mean()

        invalid_columns = missing_ratio[missing_ratio > threshold]

        if not invalid_columns.empty:
            logger.error(
                f"{dataset_name} columns exceeding missing threshold: "
                f"{invalid_columns.to_dict()}"
            )
            return False

        return True

    # =====================================================
    # STEP 4: Duplicate Record Detection
    # =====================================================
    def detect_duplicates(self, df: pd.DataFrame, dataset_name: str) -> bool:
        logger.info(f"Duplicate record check started for {dataset_name} dataset")

        id_column = self.schema["dataset"]["id_column"]
        duplicate_count = df.duplicated(subset=id_column).sum()

        if duplicate_count > 0:
            logger.error(
                f"{dataset_name} dataset contains {duplicate_count} duplicate records "
                f"based on ID column '{id_column}'"
            )
            return False

        return True

    # =====================================================
    # STEP 5: Data Drift Detection (Train vs Test)
    # =====================================================
    def perform_drift_detection(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> dict:
        logger.info("Data drift detection started (Train vs Test)")

        threshold = self.schema["drift_monitoring"]["threshold"]

        drift_report = detect_data_drift(
            base_df=train_df,
            current_df=test_df,
            threshold=threshold
        )
        logger.info("Data drift detection completed successfully (Train vs Test)")

        return drift_report

    # =====================================================
    # STEP 6: Save Validation Artifacts
    # =====================================================
 
    def save_validation_artifacts(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        drift_report: dict,
        validation_report: dict,
        overall_status: bool
    ) -> DataValidationArtifact:

        # 1️⃣ Create base validation directory
        os.makedirs(
            self.data_validation_config.data_validation_dir,
            exist_ok=True
        )

        # 2️⃣ Create valid data directory
        os.makedirs(
            self.data_validation_config.valid_data_dir,
            exist_ok=True
        )

        # 3️⃣ Save validated datasets
        valid_train_path = os.path.join(
            self.data_validation_config.valid_data_dir,
            "train.csv"
        )
        valid_test_path = os.path.join(
            self.data_validation_config.valid_data_dir,
            "test.csv"
        )

        train_df.to_csv(valid_train_path, index=False)
        test_df.to_csv(valid_test_path, index=False)

        # 4️⃣ Create reports directory
        os.makedirs(
            self.data_validation_config.reports_dir,
            exist_ok=True
        )

        # 5️⃣ Save JSON reports (directories auto-created inside write_json)
        write_json(
            self.data_validation_config.drift_report_path,
            drift_report
        )

        write_json(
            self.data_validation_config.validation_report_path,
            validation_report
        )

        # 6️⃣ Return artifact
        return DataValidationArtifact(
            # validation_status=overall_status,
            valid_train_file_path=valid_train_path,
            valid_test_file_path=valid_test_path,
            drift_report_file_path=self.data_validation_config.drift_report_path,
            # validation_report_file_path=self.data_validation_config.validation_report_path
        )

    # =====================================================
    # PIPELINE ORCHESTRATOR
    # =====================================================
    def initiate_data_validation(self) -> DataValidationArtifact:

        try:

            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.tested_file_path)
            logger.info("Data Validation pipeline started")

            validation_report = {}

            # Step 1
            schema_train = self.validate_schema(train_df, "Train")
            schema_test = self.validate_schema(test_df, "Test")
            validation_report["schema_validation"] = {
                "train": schema_train,
                "test": schema_test
            }

            # Step 2
            structure_valid = self.validate_train_test_structure(train_df, test_df)
            validation_report["train_test_structure_validation"] = structure_valid

            # Step 3
            missing_train = self.validate_missing_values(train_df, "Train")
            missing_test = self.validate_missing_values(test_df, "Test")
            validation_report["missing_value_check"] = {
                "train": missing_train,
                "test": missing_test
            }

            # Step 4
            duplicate_train = self.detect_duplicates(train_df, "Train")
            duplicate_test = self.detect_duplicates(test_df, "Test")
            validation_report["duplicate_record_check"] = {
                "train": duplicate_train,
                "test": duplicate_test
            }

            # Step 5
            drift_report = self.perform_drift_detection(train_df, test_df)
            drift_detected = any(
                col["drift_detected"] for col in drift_report.values()
            )
            validation_report["data_drift_detection"] = {
                "drift_detected": drift_detected
            }

            # Final status
            overall_status = all([
                schema_train,
                schema_test,
                structure_valid,
                missing_train,
                missing_test,
                duplicate_train,
                duplicate_test
            ])

            validation_report["overall_validation_status"] = overall_status

            dava_validation_artifact = self.save_validation_artifacts(
                train_df=train_df,
                test_df=test_df,
                drift_report=drift_report,
                validation_report=validation_report,
                overall_status=overall_status
            )

            # if not overall_status:
            #     raise CustomException(
            #         "Data Validation failed. Refer validation_report.json for details."
            #     )

            logger.info("Data Validation completed successfully")
            return dava_validation_artifact

        except Exception as e:
            logger.error("Data Validation pipeline failed", exc_info=True)
            raise CustomException(e)





def main():
    try:
        from src.entities.config.training_pipeline_config import TrainingPipelineConfig 
        from src.entities.config.data_ingestion_config import DataIngestionConfig
        from src.entities.artifact.artifacts_entity import DataIngestionAftifacts
        from src.entities.config.data_validation_config import DataValidationConfig
        # from src.entities.artifact.artifacts_entity import DataIngestionAftifacts  
        from src.common.utils import read_yaml

        config = read_yaml(os.path.join("config", "config.yaml"))

        training_pipeline_config = TrainingPipelineConfig(config=config)

        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)

        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingeted_train_path = data_ingestion_config.ingested_train_file_path
        data_ingeted_test_path = data_ingestion_config.ingested_test_file_path

        data_ingestion_artifact = DataIngestionAftifacts(data_ingeted_train_path,data_ingeted_test_path)
        
        print("abc")

        
        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)


        


    except Exception as e:
        logger.error("Data ingestion stage failed", exc_info=True)
        raise CustomException(e)



if __name__ == "__main__":
    main()