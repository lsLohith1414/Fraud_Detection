import os

from src.constants.data_ingestion_constant import (
    DATA_INGESTION_DIR_NAME,
    RAW_DATA_DIR_NAME,
    INGESTED_DATA_DIR_NAME,
    COMBINED_RAW_FILE_NAME,
)


from src.constants.training_pipeline import TRAIN_FILE_NAME, TEST_FILE_NAME

from src.entities.config.training_pipeline_config import TrainingPipelineConfig


class DataIngestionConfig:
    """
    Configuration required for data ingestion step.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        config = training_pipeline_config.config

        di = config["data_ingestion"]

        # -------------------------
        # Values from config.yaml
        # -------------------------
        self.data_root_dir: str = di["data_root_dir"]
        self.source_folders: list[str] = di["source_folders"]
        self.train_test_split_ratio: float = di["train_test_split_ratio"]

        # -------------------------
        # Artifact directories
        # -------------------------
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )

        self.raw_data_dir: str = os.path.join(
            self.data_ingestion_dir, RAW_DATA_DIR_NAME
        )

        # -------------------------
        # Artifact file paths
        # -------------------------
        self.raw_data_file_path: str = os.path.join(
            self.raw_data_dir, COMBINED_RAW_FILE_NAME
        )

        self.ingested_train_file_path: str = os.path.join(
            self.data_ingestion_dir, INGESTED_DATA_DIR_NAME, TRAIN_FILE_NAME
        )

        self.ingested_test_file_path: str = os.path.join(
            self.data_ingestion_dir, INGESTED_DATA_DIR_NAME, TEST_FILE_NAME
        )
