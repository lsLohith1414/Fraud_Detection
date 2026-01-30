# src/common/utilities.py

import yaml
import json
import pandas as pd
from scipy.stats import ks_2samp

def read_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
 
import os
import json
import numpy as np
from typing import Any


def make_json_serializable(obj: Any):
    """
    Recursively convert numpy / pandas objects into JSON-serializable
    Python native types.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def write_json(file_path: str, content: dict) -> None:
    """
    Safely write JSON by:
    1. Creating parent directories
    2. Converting numpy / pandas types
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    serializable_content = make_json_serializable(content)

    with open(file_path, "w") as f:
        json.dump(serializable_content, f, indent=4)



def detect_data_drift(
    base_df: pd.DataFrame,
    current_df: pd.DataFrame,
    threshold: float
) -> dict:

    drift_report = {}

    for col in base_df.columns:
        if base_df[col].dtype != "object":
            stat, p_value = ks_2samp(
                base_df[col].dropna(),
                current_df[col].dropna()
            )

            drift_report[col] = {
                "p_value": float(p_value),
                "drift_detected": p_value < threshold
            }

    return drift_report



import joblib
from typing import Any


def save_preprocessor(
    preprocessor: Any,
    file_path: str
) -> None:
    """
    Save a fitted preprocessor (scaler, encoder, pipeline, etc.)
    to a .pkl file.
    """

    # Create parent directory if not exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save preprocessor
    joblib.dump(preprocessor, file_path)