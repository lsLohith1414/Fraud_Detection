from datetime import datetime
import os

from src.constants.training_pipeline import ARTIFACT_DIR_NAME


class TrainingPipelineConfig:
    """
    Configuration for a single training pipeline run.
    Responsible only for artifact root creation.
    """

    def __init__(self, config: dict, timestamp: datetime = None):

        if timestamp is None:
            timestamp = datetime.now()
        
        self.config = config

        self.timestamp = timestamp.strftime("%m_%d_%Y_%H_%M")

        # # pipeline name comes from YAML
        # self.pipeline_name: str = config["project"]["name"]

        # root artifact directory (artifacts/<timestamp>)
        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR_NAME,
            self.timestamp
        )
