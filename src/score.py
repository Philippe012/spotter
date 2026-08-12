import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

from src.config import DECEMBER_CHART_FILE
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    PREDICTIONS_DIR, FIGURES_DIR, TARGET_COLUMN, ID_COLUMN,
    MIN_PREDICTED_RATE, MAX_PREDICTED_RATE
)
from src.utils import setup_logger

logger = setup_logger("score")

def validate_predictions(predictions_df: pd.DataFrame, 
                         template_df: Optional[pd.DataFrame] = None) -> bool:
    
    logger.info("Validating predictions...")
    
    required_cols = [ID_COLUMN, "predicted_rate"]
    for col in required_cols:
        if col not in predictions_df.columns:
            logger.error(f"Missing required column: {col}")
            return False
        
    if predictions_df["predicted_rate"].isnull().any():
        logger.error("Missing values found in predictions")
        return False
    
    if (predictions_df["predicted_rate"] <= 0).any():
        logger.warning("Some predictions are <= 0")
    
    if template_df is not None:
        if len(predictions_df) != len(template_df):
            logger.error(f"Prediction count ({len(predictions_df)}) doesn"t match template ({len(template_df)})")
            return False
        
        missing_ids = set(template_df[ID_COLUMN]) - set(predictions_df[ID_COLUMN])
        if missing_ids:
            logger.error(f"Missing load_ids: {missing_ids}")
            return False
    
    pred_stats = predictions_df["predicted_rate"].describe()
    logger.info(f"Prediction statistics:")
    logger.info(f"Count: {pred_stats["count"]:.0f}")
    logger.info(f"Mean: ${pred_stats["mean"]:.2f}")
    logger.info(f"Std: ${pred_stats["std"]:.2f}")
    logger.info(f"Min: ${pred_stats["min"]:.2f}")
    logger.info(f"Max: ${pred_stats["max"]:.2f}")
    
    # Checking unreasonable values
    if (predictions_df["predicted_rate"] < 50).any():
        logger.warning("Some predictions are below $50 (very low)")
    
    if (predictions_df["predicted_rate"] > 10000).any():
        logger.warning("Some predictions are above $10,000 (very high)")
    
    logger.info("Validation passed!")
    return True

def plot_december_chart(predictions_df: pd.DataFrame, 
                        decembed_data: pd.DataFrame,
                        save_path: Optional[Path] = None) -> plt.Figure:
    
    logger.info("Generating December chart...")
    
    if "date" in decembed_data.columns:
        chart_data = decembed_data.copy()
        chart_data["predicted_rate"] = predictions_df["predicted_rate"].values
        
        chart_data["date"] = pd.to_datetime(chart_data["date"])
        chart_data = chart_data.sort_values("date")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(chart_data["date"], chart_data["predicted_rate"], 
                marker="o", linewidth=2, markersize=6, color="blue")
        ax.set_xlabel("Date (December 2025)", fontsize=12)
        ax.set_ylabel("Predicted Rate ($)", fontsize=12)
        ax.set_title("December 2025 - Predicted Freight Rates", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        
        mean_rate = chart_data["predicted_rate"].mean()
        ax.axhline(y=mean_rate, color="red", linestyle="--", 
                  label=f"Mean: ${mean_rate:.2f}")
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"December chart saved to {save_path}")
        else:
            plt.show()
        
        return fig
    else:
        logger.error("Date column not found in December data")
        return None

def main(predictions_file: str = None, 
        december_file: str = None,
        chart_save_path: str = None):
    
    if predictions_file is None:
        predictions_file = PREDICTIONS_DIR / "validation_predictions.csv"
    else:
        predictions_file = Path(predictions_file)
    
    if chart_save_path is None:
        chart_save_path = FIGURES_DIR / "december_chart.png"
    else:
        chart_save_path = Path(chart_save_path)
    
    logger.info(f"Loading predictions from {predictions_file}")
    predictions_df = pd.read_csv(predictions_file)
    
    if not validate_predictions(predictions_df):
        logger.error("Prediction validation failed!")
        return False
    
    
    if december_file is None:
        december_file = DECEMBER_CHART_FILE
    else:
        december_file = Path(december_file)
    
    if december_file.exists():
        decembed_data = pd.read_csv(december_file)
        plot_december_chart(predictions_df, decembed_data, chart_save_path)
    else:
        logger.warning(f"December chart file not found at {december_file}")
    
    # Save predictions
    output_path = PREDICTIONS_DIR / "validation_predictions_formatted.csv"
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Formatted predictions saved to {output_path}")
    logger.info("Scoring complete!")
    return True

if __name__ == "__main__":
    main()