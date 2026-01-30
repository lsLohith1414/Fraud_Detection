# src/constants/training_pipeline.py

"""
Structural constants for the training pipeline.
These define internal artifact naming and directory structure.
"""

# Root artifacts directory
ARTIFACT_DIR_NAME: str = "artifacts"

# Common dataset files
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Saved models (versioned / promoted models)
SAVED_MODEL_DIR_NAME: str = "saved_models"

# Model & preprocessing artifacts
MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
