import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from src.config import (
    TRAIN_FILE, VALIDATION_FILE, TEMPLATE_FILE, 
    TARGET_COLUMN, ID_COLUMN, CATEGORICAL_FEATURES
)
from src.utils import setup_logger, validate_dataframe

logger = setup_logger("data_loader")


def load_train_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    if file_path is None:
        file_path = TRAIN_FILE
    
    logger.info(f"Loading training data from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Training data file is not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    required_cols = [TARGET_COLUMN, ID_COLUMN]
    validate_dataframe(df, required_cols)
    
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
    
    logger.info(f"Data shape: {df.shape}")
    return df

def load_validation_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    if file_path is None:
        file_path = VALIDATION_FILE
    
    logger.info(f"Loading validation data from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Validation file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    required_cols = [ID_COLUMN]
    validate_dataframe(df, required_cols)
    
    if TARGET_COLUMN in df.columns:
        logger.warning(f"Target column "{TARGET_COLUMN}" found in validation data - removing")
        df = df.drop(columns=[TARGET_COLUMN])
    
    logger.info(f"Data shape: {df.shape}")
    return df


def load_template(file_path: Optional[Path] = None) -> pd.DataFrame:
    if file_path is None:
        file_path = TEMPLATE_FILE
    
    logger.info(f"Loading template from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Template file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded template with {len(df):,} rows")
    
    required_cols = [ID_COLUMN]
    validate_dataframe(df, required_cols)
    
    return df

def load_december_chart_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    from src.config import DECEMBER_CHART_FILE
    
    if file_path is None:
        file_path = DECEMBER_CHART_FILE
    
    logger.info(f"Loading December chart data from {file_path}")
    
    if not file_path.exists():
        logger.warning(f"December chart file not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded December chart data with {len(df):,} rows")
    return df


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    saved_df = pd.read_csv(output_path)
    if len(saved_df) != len(df):
        logger.error(f"Saved file has {len(saved_df)} rows, expected {len(df)}")
    else:
        logger.info(f"Successfully saved {len(saved_df):,} rows")
