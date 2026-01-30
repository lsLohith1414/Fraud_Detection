import os
import yaml

from src.common.logger import get_logger
logger = get_logger(__name__)
from src.common.exception import CustomException
from src.common.utils import read_yaml


from src.entities.config.training_pipeline_config import TrainingPipelineConfig 
from src.entities.config.data_ingestion_config import DataIngestionConfig

from src.components.data_ingestion import DataIngestion






# ##################### Train Pipeline Config #####################
# training_pipeline_config = TrainingPipelineConfig(config=config)


# ##################### Data Ingestion ##################### 

# # -------------------- Data Ingestion Config --------------------
# data_ingestion_config = DataIngestionConfig(config=config, training_pipeline_config=training_pipeline_config)

# # -------------------- Initate Data ingestion --------------------
# data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
# data_ingestion_artifact = data_ingestion.run_data_ingestion()

# print(data_ingestion_artifact.trained_file_path)
# print(data_ingestion_artifact.tested_file_path)

test_path = os.path.join(os.getcwd(), "artifacts","01_27_2026_22_35","data_ingestion","ingested" ,"test.csv")
train_path = os.path.join(os.getcwd(), "artifacts","01_27_2026_22_35","data_ingestion","ingested" ,"train.csv")


import pandas as pd
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

schema_path = os.path.join(os.getcwd(), "config", "schema.yaml")

schema = read_yaml(schema_path)
print(schema)

expected_columns = schema["features"].keys()
print(expected_columns)
actual_columns = set(train_df.columns)
print("actual columns: ", actual_columns)





