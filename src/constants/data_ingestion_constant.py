# src/constants/data_ingestion.py

"""
Structural constants for data ingestion artifacts. 
"""

# Main ingestion directory inside artifacts/
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Sub-directories
RAW_DATA_DIR_NAME: str = "raw"
INGESTED_DATA_DIR_NAME: str = "ingested"

# File created after combining all source CSVs 
COMBINED_RAW_FILE_NAME: str = "raw_data.csv"
