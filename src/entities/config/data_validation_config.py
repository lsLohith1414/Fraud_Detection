# src/entities/config/data_validation_config.py

import os
from src.constants.data_validation_constant import (
    DATA_VALIDATION_DIR_NAME,
    VALID_DATA_DIR_NAME,
    INVALID_DATA_DIR_NAME,
    DRIFT_REPORT_FILE_NAME, 
    VALIDATION_REPORT_FILE_NAME,
    REPORT_DIR_NAME
)

from src.constants.training_pipeline import (
    TEST_FILE_NAME,
    TRAIN_FILE_NAME
)


from src.entities.config.training_pipeline_config import TrainingPipelineConfig 

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        config=training_pipeline_config.config

        dv = config["data_validation"]

        self.schema_path: str = dv["schema_path"]
        self.missing_value_threshold: float = dv["missing_value_threshold"]
        self.drift_threshold: float = dv["drift_threshold"]

        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME
        )

        self.valid_data_dir = os.path.join(
            self.data_validation_dir,
            VALID_DATA_DIR_NAME
        )

        self.invalid_data_dir = os.path.join(
            self.data_validation_dir,
            INVALID_DATA_DIR_NAME
        )

        self.reports_dir = os.path.join(
            self.data_validation_dir,
            REPORT_DIR_NAME 
        )

        self.validation_report_path = os.path.join(
            self.reports_dir,
            VALIDATION_REPORT_FILE_NAME
        )

        self.drift_report_path = os.path.join(
            self.reports_dir,
            DRIFT_REPORT_FILE_NAME
        )

        self.valid_train_file_path = os.path.join(
            self.valid_data_dir,
            TRAIN_FILE_NAME
        )

        self.valid_test_file_path = os.path.join(
            self.valid_data_dir,
            TEST_FILE_NAME
        )





