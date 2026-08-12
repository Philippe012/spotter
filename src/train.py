import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Any, Tuple, Optional
import joblib
from pathlib import Path


from src.config import (
    MODEL_PARAMS, TUNING_SPACE, CV_FOLDS, RANDOM_STATE, 
    TARGET_COLUMN, ID_COLUMN, MODEL_SAVE_PATH, 
    FEATURE_NAMES_PATH, MIN_PREDICTED_RATE, MAX_PREDICTED_RATE
)
from src.utils import setup_logger, Timer, calculate_rmse, calculate_mae, calculate_mape

logger = setup_logger('train')

class ModelTrainer:
    def __init__(self, cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.cv_results = {}
        self.best_params = None
        self.is_fitted = False
        
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           n_trials: int = 100) -> Dict[str, Any]:
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        def objective(trial):
            params = {
                    'num_leaves': trial.suggest_int('num_leaves', 2, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0)
            }
            
            params.update(MODEL_PARAMS)
            
            model = lgb.LGBMRegressor(**params, random_state=self.random_state)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            rmse = calculate_rmse(y_val, y_pred)
            
            return rmse
        
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        
        with Timer() as timer:
            study.optimize(objective, n_trials, show_progress_bar=True)
            
        logger.info(f"Hyperparameter tuning completed in {timer.seconds:.1f} seconds")
        logger.info(f"Best RMSE: {study.best_value:.2f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        self.best_params = study.best_params
        return study.best_params
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      params: Dict[str, Any]) -> Dict[str, Any]:
        # Performing cross validation with some given parameters
            logger.info(f"Starting {self.cv_folds}-fold cross-validation...")
        
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = {'rmse': [], 'mae': [], 'mape': []}
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = lgb.LGBMRegressor(**params, random_state=self.random_state)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val)
                
                cv_scores['rmse'].append(calculate_rmse(y_val, y_pred))
                cv_scores['mae'].append(calculate_mae(y_val, y_pred))
                cv_scores['mape'].append(calculate_mape(y_val, y_pred))
                
                logger.info(f"Fold {fold}: RMSE={cv_scores['rmse'][-1]:.2f}, "
                        f"MAE={cv_scores['mae'][-1]:.2f}, "
                        f"MAPE={cv_scores['mape'][-1]:.2f}%")
                
                results = {
                    'mean_rmse': np.mean(cv_scores['rmse']),
                    'std_rmse': np.std(cv_scores['rmse']),
                    'mean_mae': np.mean(cv_scores['mae']),
                    'std_mae': np.std(cv_scores['mae']),
                    'mean_mape': np.mean(cv_scores['mape']),
                    'std_mape': np.std(cv_scores['mape']),
                    'cv_scores': cv_scores
                }
                
                logger.info(f"CV Results: RMSE={results['mean_rmse']:.2f} (±{results['std_rmse']:.2f}), "
                   f"MAE={results['mean_mae']:.2f} (±{results['std_mae']:.2f})")
        
                self.cv_results = results
                return results
            
            
            def train_final_model(self, X: np.ndarray, y: np.ndarray,
                                  params: Dict[str, Any]) -> lgb.LGBMRegressor:
                
                # Training final model on all data with best parameters
                logger.info("Training final model on all data...")
        
                with Timer() as timer:
                    model = lgb.LGBMRegressor(**params, random_state=self.random_state)
                    model.fit(X, y)
                
                logger.info(f"Final model trained in {timer.seconds:.1f} seconds")
                
                self.model = model
                self.is_fitted = True
                
                self.feature_importance = pd.DataFrame({
                    'feature': range(X.shape[1]),
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return model
            
            def train(self, X: np.ndarray, y: np.ndarray, 
                tune_hyperparameters: bool = True,
                n_trials: int = 100) -> lgb.LGBMRegressor:
                
                logger.info(f"Starting model training with {len(y):,} samples, {X.shape[1]} features")
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state
                )
                
                logger.info(f"Training set: {len(y_train):,} samples")
                logger.info(f"Validation set: {len(y_val):,} samples")
                
                # Hyperpara,meter tuning
                if tune_hyperparameters:
                    params = self.tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
                else:
                    params = {
                    'num_leaves': 31,
                    'max_depth': -1,
                    'learning_rate': 0.1,
                    'n_estimators': 300,
                    'min_child_samples': 20,
                    'subsample': 1.0,
                    'colsample_bytree': 1.0,
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.0,
                    'min_split_gain': 0.0
                }
                logger.info("Using default parameters (no tuning)")
                
                params.update(MODEL_PARAMS)
                
                self.cross_validate(X, y, params)
                model = self.train_final_model(X, y, params)
                
                return model
            
            
            def save_models(self, file_path: Optional[Path] = None) -> None:
                if file_path is None:
                    file_path = MODEL_SAVE_PATH
                    
                if not self.is_fitted:
                    raise ValueError("Model must be trained before saving")
                
                file_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.model, file_path)
                logger.info(f"Model saved to {file_path}")
                
                
                if self.feature_importance is not None:
                    importance_path = file_path.parent / 'feature_importance.csv'
                    self.feature_importance.to_csv(importance_path, index=False)
                    logger.info(f"Feature importance saved to {importance_path}")
                    
            
            def load_model(self, file_path: Optional[Path] = None) -> lgb.LGBMRegressor:
                if file_path is None:
                    file_path = MODEL_SAVE_PATH
                    
                if not file_path.exists():
                    raise FileNotFoundError(f"Model file not found: {file_path}")
                
                
                self.model = joblib.load(file_path)
                self.is_fitted = True
                logger.info(f"Model loaded from {file_path}")
                return self.model
            