import os
import pandas as pd

from src.common.exception import CustomException


from src.entities.artifact.artifacts_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)

from src.entities.config.training_pipeline_config import TrainingPipelineConfig
from src.entities.config.data_validation_config import DataValidationConfig

from src.entities.config.data_transformation_config import DataTransformationConfig

# from src.entities.artifact.artifacts_entity import DataIngestionAftifacts
from src.common.utils import read_yaml


from src.common.logger import get_logger

logger = get_logger(__name__)


class DataTransformation:
    """
    BASIC CLEANING STAGE

    This stage performs:
    - Duplicate removal
    - Datatype correction
    - Basic null row cleanup
    - Schema validation

    NO feature engineering
    NO encoding
    NO scaling
    NO SMOTE
    NO model-related processing
    """

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        self.config = data_transformation_config
        self.validation_artifact = data_validation_artifact

    # ---------------------------------------------------------
    # BASIC CLEANING
    # ---------------------------------------------------------
    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1️⃣ Standardize column names
        df.columns = df.columns.str.strip()

        # 2️⃣ Remove duplicate rows
        df = df.drop_duplicates()

        # 3️⃣ Fix datetime columns
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        if "LastLogin" in df.columns:
            df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")



        return df

    # ---------------------------------------------------------
    # MAIN TRANSFORMATION ENTRY
    # ---------------------------------------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting Basic Data Cleaning Stage")

            train_df = pd.read_csv(
                self.validation_artifact.valid_train_file_path
            )
            test_df = pd.read_csv(
                self.validation_artifact.valid_test_file_path
            )

            # Apply basic cleaning
            train_df = self.basic_cleaning(train_df)
            test_df = self.basic_cleaning(test_df)

            # Ensure artifact directory exists
            os.makedirs(
                os.path.dirname(self.config.transformed_train_file_path),
                exist_ok=True,
            )

            # Save cleaned data
            train_df.to_csv(
                self.config.transformed_train_file_path,
                index=False,
            )

            test_df.to_csv(
                self.config.transformed_test_file_path,
                index=False,
            )

            logger.info("Basic Cleaning Completed Successfully")

            return DataTransformationArtifact(
                transformed_train_file_path=self.config.transformed_train_file_path,
                transformed_test_file_path=self.config.transformed_test_file_path,
                transformed_object_file_path=None,  # No preprocessor here anymore
            )

        except Exception as e:
            logger.error("Data Transformation failed", exc_info=True)
            raise CustomException(e)







def main():
    try:


        config = read_yaml(os.path.join("config", "config.yaml"))

        training_pipeline_config = TrainingPipelineConfig(config=config)

        data_validation_config = DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_valid_train_path = data_validation_config.valid_train_file_path
        data_valid_test_path = data_validation_config.valid_test_file_path
        data_valid_drift_report = data_validation_config.drift_report_path

        data_validation_artifact = DataValidationArtifact(
            valid_train_file_path=data_valid_train_path,
            valid_test_file_path=data_valid_test_path,
            drift_report_file_path=data_valid_drift_report,
        )

        data_transformation_config = DataTransformationConfig(training_pipeline_config)

        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_validation_artifact=data_validation_artifact,
        )
        data_transformation.initiate_data_transformation()

    except Exception as e:
        logger.error("Data ingestion stage failed", exc_info=True)
        raise CustomException(e)


if __name__ == "__main__":
    main()
