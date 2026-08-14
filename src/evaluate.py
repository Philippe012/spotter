import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import FIGURES_DIR, REPORTS_DIR, TARGET_COLUMN, ID_COLUMN
from src.utils import setup_logger, calculate_rmse, calculate_mae, calculate_mape

logger = setup_logger("evaluate")

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.predictions = None
        self.actuals = None
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # calculating evaluation metrics
        if np.mean(y_pred) < 10 and np.mean(y_true) > 100:
            logger.warning("Predictions appear to be in log-space. Converting to original scale...")
            y_pred = np.expm1(y_pred)
        
        metrics = {
            "rmse": calculate_rmse(y_true, y_pred),
            "mae": calculate_mae(y_true, y_pred),
            "mape": calculate_mape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
        
        errors = y_pred - y_true
        metrics.update({
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "max_error": np.max(np.abs(errors)),
            "median_error": np.median(np.abs(errors))
        })
        
        for threshold in [0.1, 0.2, 0.3, 0.5]:
            pct_within = np.mean(np.abs(errors / y_true) < threshold) * 100
            metrics[f"pct_within_{int(threshold*100)}%"] = pct_within
        
        self.metrics = metrics
        
        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[Path] = None) -> plt.Figure:
        
        residuals = y_pred - y_true
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color="red", linestyle="--")
        axes[0, 0].set_xlabel("Predicted Rate ($)")
        axes[0, 0].set_ylabel("Residuals ($)")
        axes[0, 0].set_title("Residuals vs Predicted")
        
        axes[0, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        axes[0, 1].axvline(x=0, color="red", linestyle="--")
        axes[0, 1].set_xlabel("Residuals ($)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Residuals Distribution")
        
        axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[1, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       "red", linestyle="--")
        axes[1, 0].set_xlabel("Actual Rate ($)")
        axes[1, 0].set_ylabel("Predicted Rate ($)")
        axes[1, 0].set_title("Actual vs Predicted")
        
        
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot of Residuals")
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Residual plot saved to {save_path}")
        
        return fig
    
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               feature_names: list,
                               save_path: Optional[Path] = None,
                               top_n: int = 20) -> plt.Figure:
        importance_df = feature_importance.copy()
        importance_df["feature_name"] = importance_df["feature"].apply(
            lambda x: feature_names[x] if x < len(feature_names) else f"Feature_{x}"
        )
        
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(top_features)), top_features["importance"])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature_name"])
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    
    def plot_predictions_chart(self, df: pd.DataFrame, 
                              date_col: str = "date",
                              value_col: str = "predicted_rate",
                              save_path: Optional[Path] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        df = df.copy()
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            ax.plot(df[date_col], df[value_col], marker="o", linewidth=2, markersize=4)
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted Rate ($)")
            ax.set_title("December Predictions Chart")
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"December chart saved to {save_path}")
        
        return fig
    
    
    def generate_report(self, metrics: Dict[str, float], 
                       cv_results: Optional[Dict[str, Any]] = None,
                       save_path: Optional[Path] = None) -> str:
        # Generate a text report of model performance
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("Performance Metrics:")
        report.append("-" * 40)
        for key, value in metrics.items():
            if "pct_within" in key:
                report.append(f"  {key}: {value:.2f}%")
            else:
                report.append(f"  {key}: {value:.4f}")
        report.append("")
        
        if cv_results:
            report.append("Cross-Validation Results:")
            report.append("-" * 40)
            report.append(f"Mean RMSE: {cv_results["mean_rmse"]:.4f} (+/- {cv_results["std_rmse"]:.4f})")
            report.append(f"Mean MAE: {cv_results["mean_mae"]:.4f} (+/- {cv_results["std_mae"]:.4f})")
            report.append(f"Mean MAPE: {cv_results["mean_mape"]:.2f}% (+/- {cv_results["std_mape"]:.2f}%)")
        report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        logger.info(report_text)
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text