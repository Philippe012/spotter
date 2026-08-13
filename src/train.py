import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import optuna
import xgboost as xgb
from optuna.samplers import TPESampler
from scipy.optimize import minimize
from typing import Dict, Any, Optional
import joblib
from pathlib import Path

from src.config import (
    MODEL_PARAMS, TUNING_SPACE, CV_FOLDS, RANDOM_STATE,
    TARGET_COLUMN, MODEL_SAVE_PATH,
    MIN_PREDICTED_RATE, MAX_PREDICTED_RATE
)
from src.utils import setup_logger, Timer, calculate_rmse, calculate_mae, calculate_mape

logger = setup_logger("train")


class ModelTrainer:
    def __init__(self, cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.cv_results = {}
        self.best_params = None
        self.is_fitted = False

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=100):
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", *TUNING_SPACE["num_leaves"]),
                "max_depth": trial.suggest_int("max_depth", *TUNING_SPACE["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *TUNING_SPACE["learning_rate"], log=True),
                "n_estimators": trial.suggest_int("n_estimators", *TUNING_SPACE["n_estimators"]),
                "min_child_samples": trial.suggest_int("min_child_samples", *TUNING_SPACE["min_child_samples"]),
                "subsample": trial.suggest_float("subsample", *TUNING_SPACE["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *TUNING_SPACE["colsample_bytree"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *TUNING_SPACE["reg_alpha"]),
                "reg_lambda": trial.suggest_float("reg_lambda", *TUNING_SPACE["reg_lambda"]),
                "min_split_gain": trial.suggest_float("min_split_gain", *TUNING_SPACE["min_split_gain"]),
            }
            params.update(MODEL_PARAMS)

            local_params = params.copy()
            local_params.pop("random_state", None)
            model = lgb.LGBMRegressor(**local_params, random_state=self.random_state)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            y_pred = model.predict(X_val)
            return calculate_rmse(y_val, y_pred)

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        with Timer() as timer:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info(f"Hyperparameter tuning completed in {timer.seconds:.1f} seconds")
        logger.info(f"Best RMSE: {study.best_value:.2f}")
        logger.info(f"Best parameters: {study.best_params}")
        self.best_params = study.best_params
        return study.best_params

    def cross_validate(self, X, y, params):
        logger.info(f"Starting {self.cv_folds}-fold cross-validation...")
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores = {"rmse": [], "mae": [], "mape": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            local_params = params.copy()
            local_params.pop("random_state", None)
            model = lgb.LGBMRegressor(**local_params, random_state=self.random_state)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            y_pred = model.predict(X_val)

            cv_scores["rmse"].append(calculate_rmse(y_val, y_pred))
            cv_scores["mae"].append(calculate_mae(y_val, y_pred))
            cv_scores["mape"].append(calculate_mape(y_val, y_pred))

            logger.info(
                f"Fold {fold}: RMSE={cv_scores['rmse'][-1]:.2f}, "
                f"MAE={cv_scores['mae'][-1]:.2f}, "
                f"MAPE={cv_scores['mape'][-1]:.2f}%"
            )

        results = {
            "mean_rmse": float(np.mean(cv_scores["rmse"])),
            "std_rmse": float(np.std(cv_scores["rmse"])),
            "mean_mae": float(np.mean(cv_scores["mae"])),
            "std_mae": float(np.std(cv_scores["mae"])),
            "mean_mape": float(np.mean(cv_scores["mape"])),
            "std_mape": float(np.std(cv_scores["mape"])),
            "cv_scores": cv_scores,
        }

        self.cv_results = results
        return results

    def train_final_model(self, X, y, params):
        logger.info("Training final model on all data...")
        with Timer() as timer:
            local_params = params.copy()
            local_params.pop("random_state", None)
            model = lgb.LGBMRegressor(**local_params, random_state=self.random_state)
            model.fit(X, y)

        logger.info(f"Final model trained in {timer.seconds:.1f} seconds")
        self.model = model
        self.is_fitted = True
        self.feature_importance = pd.DataFrame({
            "feature": range(X.shape[1]),
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return model

    def train(self, X_train, y_train, X_val, y_val, tune_hyperparameters=True, n_trials=100):
        if tune_hyperparameters:
            params = self.tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
        else:
            params = {
                "num_leaves": 31,
                "max_depth": -1,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "min_split_gain": 0.0,
            }

        params.update(MODEL_PARAMS)
        self.cross_validate(X_train, y_train, params)
        return self.train_final_model(X_train, y_train, params)

    def save_models(self, file_path: Optional[Path] = None) -> None:
        if file_path is None:
            file_path = MODEL_SAVE_PATH
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, file_path)
        logger.info(f"Model saved to {file_path}")

        if self.feature_importance is not None:
            importance_path = file_path.parent / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")

    def load_model(self, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = MODEL_SAVE_PATH
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        self.model = joblib.load(file_path)
        self.is_fitted = True
        logger.info(f"Model loaded from {file_path}")
        return self.model


class EnsembleTrainer:
    def __init__(self, cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.lgb_model = None
        self.xgb_model = None
        self.weights = np.array([0.6, 0.4], dtype=float)
        self.is_fitted = False
        self.cv_results = {}
        self.feature_importance = None

    def _default_lgb_params(self):
        return {
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_split_gain": 0.0,
        }

    def _default_xgb_params(self):
        return {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
            "tree_method": "hist",
        }

    def _optimize_weights(self, lgb_pred, xgb_pred, y_true):
        def objective(weights):
            weights = np.clip(weights, 0.05, 0.95)
            weights = weights / weights.sum()
            blended = self.weights[0] * lgb_pred + self.weights[1] * xgb_pred
            return np.sqrt(np.mean((blended - y_true) ** 2))

        result = minimize(
            objective,
            x0=np.array([0.6, 0.4], dtype=float),
            bounds=[(0.05, 0.95), (0.05, 0.95)],
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            method="SLSQP",
        )

        if not result.success:
            weights = np.array([0.6, 0.4], dtype=float)
        else:
            weights = np.clip(result.x, 0.05, 0.95)
            weights = weights / weights.sum()

        self.weights = weights
        logger.info(f"Optimal ensemble weights: LightGBM={weights[0]:.3f}, XGBoost={weights[1]:.3f}")
        return weights

    def _fit_lightgbm(self, X_train, y_train, X_val, y_val, params):
        local_params = dict(params)
        local_params.update(MODEL_PARAMS)
        local_params.pop("random_state", None)
        model = lgb.LGBMRegressor(**local_params, random_state=self.random_state)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return model

    def _fit_xgboost(self, X_train, y_train, X_val, y_val):
        params = self._default_xgb_params()
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False,
        )
        return model

    def cross_validate(self, X, y, params):
        logger.info(f"Starting ensemble cross-validation with {self.cv_folds} folds...")
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores = {"rmse": [], "mae": [], "mape": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            lgb_model = self._fit_lightgbm(X_train, y_train, X_val, y_val, params)
            xgb_model = self._fit_xgboost(X_train, y_train, X_val, y_val)

            lgb_pred = lgb_model.predict(X_val)
            xgb_pred = xgb_model.predict(X_val)
            weights = self._optimize_weights(lgb_pred, xgb_pred, y_val)
            blended = weights[0] * lgb_pred + weights[1] * xgb_pred

            cv_scores["rmse"].append(calculate_rmse(y_val, blended))
            cv_scores["mae"].append(calculate_mae(y_val, blended))
            cv_scores["mape"].append(calculate_mape(y_val, blended))

            logger.info(
                f"Fold {fold}: RMSE={cv_scores['rmse'][-1]:.2f}, "
                f"MAE={cv_scores['mae'][-1]:.2f}, "
                f"MAPE={cv_scores['mape'][-1]:.2f}%"
            )

        results = {
            "mean_rmse": float(np.mean(cv_scores["rmse"])),
            "std_rmse": float(np.std(cv_scores["rmse"])),
            "mean_mae": float(np.mean(cv_scores["mae"])),
            "std_mae": float(np.std(cv_scores["mae"])),
            "mean_mape": float(np.mean(cv_scores["mape"])),
            "std_mape": float(np.std(cv_scores["mape"])),
            "cv_scores": cv_scores,
        }
        self.cv_results = results
        return results

    def train(self, X_train, y_train, X_val, y_val, tune_hyperparameters=True, n_trials=100):
        logger.info("Training ensemble model (LightGBM + XGBoost)...")

        lgb_trainer = ModelTrainer(cv_folds=self.cv_folds, random_state=self.random_state)
        if tune_hyperparameters:
            params = lgb_trainer.tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
        else:
            params = self._default_lgb_params()

        self.cross_validate(X_train, y_train, params)

        self.lgb_model = self._fit_lightgbm(X_train, y_train, X_val, y_val, params)
        self.xgb_model = self._fit_xgboost(X_train, y_train, X_val, y_val)

        lgb_pred = self.lgb_model.predict(X_val)
        xgb_pred = self.xgb_model.predict(X_val)
        self.weights = self._optimize_weights(lgb_pred, xgb_pred, y_val)

        self.feature_importance = pd.DataFrame({
            "feature": range(X_train.shape[1]),
            "importance": self.lgb_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        self.model = {
            "lgb_model": self.lgb_model,
            "xgb_model": self.xgb_model,
            "weights": self.weights,
        }
        self.is_fitted = True
        return self.model

    def predict(self, X):
        if self.model is None:
            raise ValueError("Ensemble model has not been trained yet.")
        
        lgb_pred = self.model["lgb_model"].predict(X)
        xgb_pred = self.model["xgb_model"].predict(X) 
        
        blended = self.weights[0] * lgb_pred + self.weights[1] * xgb_pred
        return blended

    def save_models(self, file_path: Optional[Path] = None) -> None:
        if file_path is None:
            file_path = MODEL_SAVE_PATH
        if self.model is None:
            raise ValueError("Ensemble model must be trained before saving")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, file_path)
        logger.info(f"Ensemble model saved to {file_path}")

        if self.feature_importance is not None:
            importance_path = file_path.parent / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")

    def load_model(self, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = MODEL_SAVE_PATH
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        self.model = joblib.load(file_path)
        self.lgb_model = self.model["lgb_model"]
        self.xgb_model = self.model["xgb_model"]
        self.weights = np.asarray(self.model["weights"], dtype=float)
        self.is_fitted = True
        logger.info(f"Ensemble model loaded from {file_path}")
        return self.model