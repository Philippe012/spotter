from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_FILE = RAW_DATA_DIR / "train-test.csv"
VALIDATION_FILE = RAW_DATA_DIR / "validation.csv"
TEMPLATE_FILE = RAW_DATA_DIR / "validation-predictions-template.csv"
DECEMBER_CHART_FILE = RAW_DATA_DIR / "december-chart-inputs.csv"

PROCESSED_TRAIN = PROCESSED_DATA_DIR / "train_processed.parquet"
PROCESSED_VALIDATION = PROCESSED_DATA_DIR / "validation_processed.parquet"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"


SCORER_RESULTS_DIR = PROJECT_ROOT / 'scorer_results'
DECEMBER_CHART_OUTPUT = SCORER_RESULTS_DIR / 'candidate_december.png'


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODEL_DIR / "lightgbm_model.pkl"
PREPROCESSOR_SAVE_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, PREDICTIONS_DIR, FIGURES_DIR,
                 REPORTS_DIR, MODEL_DIR, SCORER_RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    
    
    
MODEL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42
}


CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2


HOLIDAY_COUNTRY = "US"
HOLIDAY_STATE = None
DATE_FEATURES = ["year", "month", "day", "dayofweek", "quarter", "dayofyear"]


CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery"]
NUMERICAL_FEATURES = ["distance", "weight", "market_index", "quote_signal"]

TARGET_COLUMN = "posted_rate"
ID_COLUMN = "load_id"

DECEMBER_FIXED = {
    'pickup': 'Lexington',
    'delivery': 'Fort Wayne',
    'distance': 360.0,
    'equipment': 'Dry Van',
    'weight': 32000
}

TUNING_SPACE = {
    "num_leaves": (16, 64),
    "max_depth": (4, 12),
    "learning_rate": (0.01, 0.1),
    "n_estimators": (200, 1000),
    "min_child_samples": (10, 50),
    "subsample": (0.7, 1.0),
    "colsample_bytree": (0.7, 1.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.0, 5.0),
    "min_split_gain": (0.0, 0.5)
}

MIN_PREDICTED_RATE = 50.0
MAX_PREDICTED_RATE = 10000.0