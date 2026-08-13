import sys
import subprocess
import argparse
from pathlib import Path
import joblib
import pandas as pd
import time

sys.path.append(str(Path(__file__).parent))

from src.config import (
    TRAIN_FILE, VALIDATION_FILE, TEMPLATE_FILE, DECEMBER_CHART_FILE,
    PREDICTIONS_DIR, FIGURES_DIR, SCORER_RESULTS_DIR,
    TARGET_COLUMN, ID_COLUMN, MODEL_SAVE_PATH, FEATURE_NAMES_PATH, REPORTS_DIR, DECEMBER_FIXED
)

from src.data_loader import load_train_data, load_validation_data, load_template, save_predictions
from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer
from src.train import ModelTrainer
from src.predict import Predictor
from src.evaluate import ModelEvaluator
from score import validate_predictions, save_december_chart
from src.utils import setup_logger, Timer

logger = setup_logger('main')


def parse_args():
    parser = argparse.ArgumentParser(description='Spotter ML Assessment Pipeline')
    parser.add_argument('--skip-tune', action='store_true', 
                       help='Skip hyperparameter tuning (use default params)')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of Optuna trials for tuning')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip training (use existing model)')
    parser.add_argument('--predict-only', action='store_true',
                       help='Only generate predictions (assumes model exists)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Starting Spotter freight rate prediction pipeline")
    
    if not TRAIN_FILE.exists():
        logger.error("Training file not found: %s", TRAIN_FILE) 
        logger.info("Expected train-test.csv in data/raw/") 
        return
    
    if not VALIDATION_FILE.exists():
        logger.error(f"Validation file not found: %s", VALIDATION_FILE)
        logger.info("Expected validation.csv data/raw/")
        return
    
    try:
        with Timer() as total_timer:
            logger.info("Loading Data")
            
            train_df = load_train_data()
            validation_df = load_validation_data()
            template_df = load_template()
            
            logger.info( "Training data: %s rows, %s columns", f"{len(train_df):,}", len(train_df.columns), ) 
            logger.info( "Validation data: %s rows, %s columns", f"{len(validation_df):,}", len(validation_df.columns), ) 
            logger.info("Template: %s rows", f"{len(template_df):,}")
            
            logger.info("Preprocessing Data")
            preprocessor = DataPreprocessor()
            
            train_processed = preprocessor.fit_transform(train_df)
            train_processed = (
                train_processed.sort_values("date").reset_index(drop=True)
            )

            logger.info(f"Training data processed: {train_processed.shape}")

            validation_processed = preprocessor.transform(validation_df)
            logger.info(f"Validation data processed: {validation_processed.shape}")
           
            
            logger.info("Feature Engineering")            
            feature_engineer = FeatureEngineer()
            
            
            train_features, feature_names = feature_engineer.fit_transform(train_processed)
                        
            logger.info(f"Training features: {train_features.shape}")
            logger.info(f"Number of features: {len(feature_names)}")

            feature_columns = [
                col for col in feature_names
                if col in train_features.columns
                and pd.api.types.is_numeric_dtype(train_features[col])
                and col != "rate_per_mile"
            ]
            
            
            train_features = train_features.sort_values("date").reset_index(drop=True)

            cutoff_index = max(1, int(len(train_features) * 0.8))
            cutoff_date = train_features["date"].iloc[cutoff_index - 1]

            train_mask = train_features["date"] <= cutoff_date
            val_mask = train_features["date"] > cutoff_date

            X_train = train_features.loc[train_mask, feature_columns].values
            y_train = train_features.loc[train_mask, TARGET_COLUMN].values

            X_time_val = train_features.loc[val_mask, feature_columns].values
            y_time_val = train_features.loc[val_mask, TARGET_COLUMN].values

            logger.info(
                "Time-based split: training=%d, validation=%d",
                len(y_train),
                len(y_time_val),
            )
            
            validation_features = feature_engineer.transform(validation_processed)

            logger.info(
                f"Validation features: {validation_features.shape}"
            )

            logger.info(f"Training matrix: {X_train.shape}")
            logger.info(f"Training targets: {y_train.shape}")
            logger.info(f"November validation matrix: {X_time_val.shape}")
            logger.info(f"November validation targets: {y_time_val.shape}")
            
            
            logger.info("Training Model")
            
            trainer = ModelTrainer()
            
            if not args.predict_only:
                model = trainer.train(
                    X_train, y_train,X_val=X_time_val, y_val=y_time_val,
                    tune_hyperparameters=not args.skip_tune,
                    n_trials=args.n_trials
                )
                
                trainer.save_models()
                joblib.dump(feature_names, FEATURE_NAMES_PATH)
                logger.info(f"Feature names saved to {FEATURE_NAMES_PATH}")
            else:
                trainer.load_model()
                logger.info("Using existing model")
            
            
            logger.info("Evaluating Model")
            
            evaluator = ModelEvaluator()
            
            y_pred_time_val = trainer.model.predict(X_time_val)

            metrics = evaluator.evaluate(
                y_time_val, y_pred_time_val
            )

            logger.info(
                "November validation performance: "
                f"RMSE=${metrics['rmse']:.2f}, "
                f"MAE=${metrics['mae']:.2f}, "
                f"R2={metrics['r2']:.4f}, "
                f"MAPE={metrics['mape']:.2f}%"
            )
            
            # Residual ploting
            evaluator.plot_residuals(
                y_time_val, y_pred_time_val,
                save_path=FIGURES_DIR / 'residual_plot.png'
            )
            
            if trainer.feature_importance is not None:
                evaluator.plot_feature_importance(
                    trainer.feature_importance,
                    feature_names,
                    save_path=FIGURES_DIR / 'feature_importance.png'
                )
            
            evaluator.generate_report(
                metrics,
                trainer.cv_results,
                save_path=REPORTS_DIR / 'model_report.txt'
            )
            
            logger.info("Generating Predictions")
            
            predictor = Predictor(trainer.model)
            
            missing_features = [
                col for col in feature_columns
                if col not in validation_features.columns
            ]

            if missing_features:
                raise ValueError(
                    f"Validation data is missing features: {missing_features}"
                )

            X_val = validation_features[feature_columns].values
            
            validation_predictions = predictor.predict_validation_data(
                X_val,
                validation_features[ID_COLUMN].values,
            )
            
            
            output_path = PREDICTIONS_DIR / 'validation_predictions.csv'
            predictor.save_predictions(validation_predictions, output_path)
            
            template_predictions = validation_predictions[
                [ID_COLUMN, "predicted_rate"]].copy()

            template_df = template_df.drop(
                columns=["predicted_rate"], errors="ignore"
            )

            template_df = template_df.merge(
                template_predictions,
                on=ID_COLUMN, how="left", validate="one_to_one"
            )
            
            template_output = PREDICTIONS_DIR / 'validation_predictions_template_filled.csv'
            predictor.save_predictions(template_df, template_output)
            
            
            
            logger.info("Generating December Chart")
            if DECEMBER_CHART_FILE.exists():
                december_data = pd.read_csv(DECEMBER_CHART_FILE)                
                december_dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
                
                december_data = pd.DataFrame({
                    "pickup": [DECEMBER_FIXED["pickup"]] * 31,
                    "delivery": [DECEMBER_FIXED["delivery"]] * 31,
                    "distance": [DECEMBER_FIXED["distance"]] * 31,
                    "equipment": [DECEMBER_FIXED["equipment"]] * 31,
                    "weight": [DECEMBER_FIXED["weight"]] * 31,
                    "market_index": [1.0] * 31,
                    "quote_signal": [0.0] * 31,
                    "pickup_lat": [39.0] * 31,
                    "delivery_lat": [41.0] * 31,
                    "pickup_lon": [-84.0] * 31,
                    "delivery_lon": [-85.0] * 31,
                    "date": december_dates,
                })
                
                december_processed = preprocessor.transform(december_data)
                december_features = feature_engineer.transform(december_processed)
                
                for col in feature_columns:
                    if col not in december_features.columns:
                        december_features[col] = 0.0

                X_dec = december_features[feature_columns].values
                
                december_predictions_df = predictor.generate_december_predictions(X_dec, december_dates)
                dec_output = DECEMBER_CHART_FILE
                dec_output = PREDICTIONS_DIR / "december_predictions.csv"
                predictor.save_predictions(december_predictions_df, dec_output)
                
                logger.info("Running with all from scores")
                
                scorer_cmd = [
                    sys.executable,
                    str(Path(__file__).parent / 'score.py'),
                    '--predictions', str(output_path),
                    '--december-predictions', str(dec_output)
                ]
                
                logger.info(f"Running: {' '.join(scorer_cmd)}")
                result = subprocess.run(scorer_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Score.py validation passed!")
                    logger.info(result.stdout)
                
                chart_path = SCORER_RESULTS_DIR / 'candidate_december.png'
                if chart_path.exists():
                    logger.info(f"December chart created: {chart_path}")
                else:
                    logger.warning("December chart not found in expected location")
            else:
                logger.error("Score.py validation failed!")
                logger.error(result.stderr)
                logger.error("Please check your prediction files")
                
            logger.info("Pipeline Completed")
            logger.info(f"Model performance:")
            logger.info(f"RMSE: ${metrics['rmse']:.2f}")
            logger.info(f"MAE: ${metrics['mae']:.2f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            logger.info(f"MAPE: {metrics['mape']:.2f}%")
            logger.info(f"Output files:")
            logger.info(f"Predictions: {output_path}")
            logger.info(f"December chart: {FIGURES_DIR / 'december_chart.png'}")
            logger.info(f"Model: {MODEL_SAVE_PATH}")

        total_timer.end = time.time()
        total_timer.seconds = total_timer.end - total_timer.start
        total_timer.milliseconds = total_timer.seconds * 1000

        logger.info(f"Total execution time: {total_timer.seconds:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()