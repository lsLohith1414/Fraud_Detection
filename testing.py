import os
import yaml

from src.common.logger import get_logger
logger = get_logger(__name__)
from src.common.exception import CustomException
from src.common.utils import read_yaml


from src.entities.config.training_pipeline_config import TrainingPipelineConfig 
from src.entities.config.data_ingestion_config import DataIngestionConfig
from src.entities.config.data_validation_config import DataValidationConfig

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation





config_path = os.path.join(os.getcwd(), "config", "config.yaml")

config = read_yaml(config_path)

##################### Train Pipeline Config #####################
training_pipeline_config = TrainingPipelineConfig(config=config)


##################### Data Ingestion ##################### 

# -------------------- Data Ingestion Config --------------------
data_ingestion_config = DataIngestionConfig(config=config, training_pipeline_config=training_pipeline_config)

# -------------------- Initate Data ingestion --------------------
data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
data_ingestion_artifact = data_ingestion.run_data_ingestion()

print(data_ingestion_artifact.trained_file_path)
print(data_ingestion_artifact.tested_file_path)


##################### Data validation #####################

# -------------------- Data Validation Config --------------------
data_validation_config = DataValidationConfig(config= config, training_pipeline_config=training_pipeline_config)

# -------------------- Initate Data validation --------------------
data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
data_validation_artifact = data_validation.initiate_data_validation()
print(data_validation_artifact)


 ##################### Data Transformation #####################
from src.entities.config.data_transformation_config import DataTransformationConfig
from src.components.data_transformation import DataTransformation

data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)

# -------------------- Initiate Data Transformation --------------------
data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
data_transformation_artifact = data_transformation.initiate_data_transformation()


##################### Model trainer #####################
from src.entities.config.model_trainer_config import ModelTrainerConfig
from src.components.model_trainer import  ModelTrainer

model_training_config = ModelTrainerConfig(training_pipeline_config)

# -------------------- Initiate Model trainer --------------------
model_trainier = ModelTrainer(model_trainer_config=model_training_config,data_transformation_artifact=data_transformation_artifact)
model_trainier.initiate_model_trainer()








