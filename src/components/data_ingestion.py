import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split



from src.common.logger import get_logger
logger = get_logger(__name__)
from src.common.exception import CustomException
from src.entities.config.data_ingestion_config import DataIngestionConfig 
from src.entities.artifact.artifacts_entity import DataIngestionAftifacts 



class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def load_source_tables(
        self,
        data_root_dir: str,
        source_folders: list[str]
    ) -> dict[str, pd.DataFrame]:
        """
        Discover and load all CSV files from configured source folders.
        """
        try:
            tables = {}

            logger.info(
                f"Starting CSV discovery under root directory: {data_root_dir}"
            )

            for folder in source_folders:
                folder_path = os.path.join(data_root_dir, folder)

                if not os.path.exists(folder_path):
                    logger.warning(
                        f"Source folder not found, skipping: {folder_path}"
                    )
                    continue

                csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

                if not csv_files:
                    logger.warning(
                        f"No CSV files found inside folder: {folder_path}"
                    )
                    continue

                for csv_path in csv_files:
                    table_name = os.path.splitext(
                        os.path.basename(csv_path)
                    )[0]

                    logger.info(
                        f"Loading source table '{table_name}' from {csv_path}"
                    )
                    tables[table_name] = pd.read_csv(csv_path)

            logger.info(
                f"Successfully loaded {len(tables)} source tables"
            )

            return tables

        except Exception as e:
            logger.error(
                "Failed while loading source tables",
                exc_info=True
            )
            raise CustomException(e)

    def build_training_dataset(
        self,
        tables: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Build final training dataset using SQL-style joins.
        """
        try:
            logger.info("Starting dataset merge using relational joins")

            df = tables["transaction_records"].copy()

            df = df.merge(tables["fraud_indicators"], on="TransactionID", how="left")
            df = df.merge(tables["transaction_metadata"], on="TransactionID", how="left")
            df = df.merge(tables["amount_data"], on="TransactionID", how="left")
            df = df.merge(tables["anomaly_scores"], on="TransactionID", how="left")
            df = df.merge(tables["transaction_category_labels"], on="TransactionID", how="left")
            df = df.merge(tables["merchant_data"], on="MerchantID", how="left")
            df = df.merge(tables["customer_data"], on="CustomerID", how="left")
            df = df.merge(tables["account_activity"], on="CustomerID", how="left")
            df = df.merge(tables["suspicious_activity"], on="CustomerID", how="left")

            logger.info(
                f"Dataset merge completed successfully. Final shape: {df.shape}"
            )


            return df

        except Exception as e:
            logger.error(
                "Failed while merging relational dataset",
                exc_info=True
            )
            raise CustomException(e)

    def split_and_persist_data(self, dataframe: pd.DataFrame):
        """
        Split dataset into train and test sets and persist artifacts.
        """
        try:
            logger.info(
                "Persisting raw combined dataset to artifact store"
            )

            os.makedirs(
                self.data_ingestion_config.raw_data_dir,
                exist_ok=True
            )

            dataframe.to_csv(
                self.data_ingestion_config.raw_data_file_path,
                index=False
            )

            train_df, test_df = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            ingested_dir = os.path.dirname(
                self.data_ingestion_config.ingested_train_file_path
            )
            os.makedirs(ingested_dir, exist_ok=True)

            train_df.to_csv(
                self.data_ingestion_config.ingested_train_file_path,
                index=False
            )
            test_df.to_csv(
                self.data_ingestion_config.ingested_test_file_path,
                index=False
            )

            logger.info(
                "Train-test split completed and artifacts saved successfully"
            )

        except Exception as e:
            logger.error(
                "Failed while splitting and saving dataset",
                exc_info=True
            )
            raise CustomException(e)

    def run_data_ingestion(self) -> DataIngestionAftifacts:
        """
        Orchestrates the complete data ingestion process.
        """
        try:
            logger.info(
                "Data Ingestion started"
            )

            tables = self.load_source_tables(
                data_root_dir=self.data_ingestion_config.data_root_dir,
                source_folders=self.data_ingestion_config.source_folders
            )

            dataframe = self.build_training_dataset(tables)

            self.split_and_persist_data(dataframe)

            artifact = DataIngestionAftifacts(
                trained_file_path=self.data_ingestion_config.ingested_train_file_path,
                tested_file_path=self.data_ingestion_config.ingested_test_file_path
            )

            logger.info(
                "Data Ingestion completed successfully"
            )

            return artifact

        except Exception as e:
            logger.error(
                "Data ingestion pipeline failed",
                exc_info=True
            )
            raise CustomException(e)









##################### Train Pipeline Config #####################



##################### Data Ingestion ##################### 

# -------------------- Data Ingestion Config --------------------


# -------------------- Initate Data ingestion --------------------




def main():
    try:
        from src.entities.config.training_pipeline_config import TrainingPipelineConfig 
        # from src.entities.artifact.artifacts_entity import DataIngestionAftifacts  
        from src.common.utils import read_yaml

        config = read_yaml(os.path.join("config", "config.yaml"))

        training_pipeline_config = TrainingPipelineConfig(config=config)

        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)

        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

        data_ingestion_artifact = data_ingestion.run_data_ingestion()


    except Exception as e:
        logger.error("Data ingestion stage failed", exc_info=True)
        raise CustomException(e)



if __name__ == "__main__":
    main()


