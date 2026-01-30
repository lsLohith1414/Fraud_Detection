import os
import yaml


from src.common.logger import get_logger
logger = get_logger(__name__)
from src.common.exception import CustomException
from src.common.utils import read_yaml

from src.constants.data_transformation_constant import (
    DATA_TRANSFORMATION_DIR_NAME , # "data_transformation"
    TRANSFORMED_DATA_DIR ,# "transformed"
    TRANSFORMED_TRAIN_FILE_NAME ,# "train.npy"
    TRANSFORMED_TEST_FILE_NAME , # = "test.npy"
    PREPROCESSOR_DIR ,# = "preprocessor"
    PREPROCESSOR_FILE_NAME ,# = "preprocessor.pkl"

)

from src.entities.config.training_pipeline_config import TrainingPipelineConfig




class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        config = training_pipeline_config.config 
        self.numerical_scaler = config["data_transformation"]["numerical_scaler"]

        self.categorical_encoder = config["data_transformation"]["categorical_encoder"]


        # paths to store in artifacts

        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, # 
                                                    DATA_TRANSFORMATION_DIR_NAME )
        
        self.transformed_data_dir = os.path.join(self.data_transformation_dir,TRANSFORMED_DATA_DIR)

        self.transformed_train_file_path = os.path.join(self.transformed_data_dir,TRANSFORMED_TRAIN_FILE_NAME) #

        self.transformed_test_file_path = os.path.join(self.transformed_data_dir,TRANSFORMED_TEST_FILE_NAME) # 

        self.preprocessor_dir = os.path.join(self.data_transformation_dir,PREPROCESSOR_DIR)

        self.preprocessor_file_path = os.path.join(self.preprocessor_dir,PREPROCESSOR_FILE_NAME)  # 


