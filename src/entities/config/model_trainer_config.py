import os
from src.constants.model_trainer_constants import  (
    MODEL_TRAINER_DIR_NAME,
    TRAINED_MODEL_FILE_NAME,
    TRAINED_MODEL_DIR_NAME

)

from src.entities.config.training_pipeline_config import TrainingPipelineConfig


from src.common.utils import read_yaml



class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        config = training_pipeline_config.config

        mt_cfg = config["model_training"]

        self.model_name = mt_cfg["model_registry"]["model_name"]
        self.model_alias = mt_cfg["model_registry"]["model_alias"]
        self.target_column = mt_cfg["target_column"]


        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            MODEL_TRAINER_DIR_NAME
        )


        self.trained_model_path = os.path.join(
            self.model_trainer_dir,
            TRAINED_MODEL_DIR_NAME,
            TRAINED_MODEL_FILE_NAME
        )




