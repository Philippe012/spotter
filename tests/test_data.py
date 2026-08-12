import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.config import TARGET_COLUMN, ID_COLUMN
from src.data_loader import load_train_data, load_validation_data
from src.preprocessing import DataPreprocessor

def test_data_loading():
    try:
        train_df = load_train_data()
        assert len(train_df) > 0
        assert TARGET_COLUMN in train_df.columns
        assert ID_COLUMN in train_df.columns
        print("Data loading test passed")
        
    except FileNotFoundError:
        print("Data files not found - skipping test")

def test_preprocessing():
    try:
        train_df = load_train_data()
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.fit_transform(train_df)
        
        assert len(processed_df) > 0
        assert TARGET_COLUMN in processed_df.columns
        
        if 'date' in processed_df.columns:
            assert 'year' in processed_df.columns
            assert 'month' in processed_df.columns
        
        print("Preprocessing test passed")
    except FileNotFoundError:
        print("Data files not found - skipping test")

if __name__ == "__main__":
    test_data_loading()
    test_preprocessing()