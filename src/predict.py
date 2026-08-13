import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import joblib

from src.config import (
    MODEL_SAVE_PATH, ID_COLUMN,
    MIN_PREDICTED_RATE, MAX_PREDICTED_RATE
)
from src.utils import setup_logger, clip_predictions

logger = setup_logger("predict")


class Predictor:
    def __init__(self, model=None):
        self.model = model
        self.is_loaded = False

    def load_model(self, model_path: Optional[Path] = None) -> None:
        if model_path is None:
            model_path = MODEL_SAVE_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(model_path)
        self.is_loaded = True
        logger.info(f"Model loaded from {model_path}")

    def _blend_predictions(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.model, dict):
            ensemble = self.model
            lgb_model = ensemble.get("lgb_model")
            xgb_model = ensemble.get("xgb_model")
            weights = np.asarray(ensemble.get("weights", [0.6, 0.4]), dtype=float)

            if lgb_model is None or xgb_model is None:
                raise ValueError("Ensemble model is missing LightGBM or XGBoost model components")

            lgb_pred = lgb_model.predict(X)
            xgb_pred = xgb_model.predict(X)
            blended = weights[0] * lgb_pred + weights[1] * xgb_pred
            return np.asarray(blended, dtype=float)

        if self.model is None and not self.is_loaded:
            raise ValueError("No model loaded. Call load_model() first")

        return np.asarray(self.model.predict(X), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self._blend_predictions(X) 
        predictions = np.expm1(predictions)
        
        predictions = clip_predictions(predictions, MIN_PREDICTED_RATE, MAX_PREDICTED_RATE)
        predictions = np.maximum(predictions, 50.0)
        return predictions

    def predict_validation_data(self, X: np.ndarray, load_ids: Union[pd.Series, list]) -> pd.DataFrame:
        logger.info(f"Generating predictions for {len(load_ids):,} validation loads...")
        predictions = self.predict(X)

        result_df = pd.DataFrame({
            ID_COLUMN: load_ids,
            "predicted_rate": predictions.round(2)
        })

        logger.info(f"Prediction summary:")
        logger.info(f"Min rate: ${predictions.min():.2f}")
        logger.info(f"Max rate: ${predictions.max():.2f}")
        logger.info(f"Mean rate: ${predictions.mean():.2f}")
        logger.info(f"Std rate: ${predictions.std():.2f}")

        return result_df

    def generate_december_predictions(self, X: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
        logger.info(f"Generating predictions for December for {len(dates)} days...")
        predictions = self.predict(X)

        december_df = pd.DataFrame({
            "pickup": ["Lexington"] * len(dates),
            "delivery": ["Fort Wayne"] * len(dates),
            "distance": [360.0] * len(dates),
            "equipment": ["Dry Van"] * len(dates),
            "weight": [32000.0] * len(dates),
            "date": dates,
            "predicted_rate": predictions.round(2)
        })

        december_df = december_df[
            ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
        ]
        return december_df

    def save_predictions(self, df: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")