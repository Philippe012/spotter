import sys
import argparse
from pathlib import Path
import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.config import (
    TRAIN_FILE, VALIDATION_FILE, TEMPLATE_FILE, DECEMBER_CHART_FILE,
    PROCESSED_TRAIN, PROCESSED_VALIDATION, PREDICTIONS_DIR, FIGURES_DIR,
    TARGET_COLUMN, ID_COLUMN, MODEL_SAVE_PATH, FEATURE_NAMES_PATH,
    MIN_PREDICTED_RATE, MAX_PREDICTED_RATE, REPORTS_DIR
)

from src.data_loader import load_train_data, load_validation_data, load_template, save_predictions
from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer
from src.train import ModelTrainer
from src.predict import Predictor
from src.evaluate import ModelEvaluator
from src.score import validate_predictions, plot_december_chart
from src.utils import setup_logger, Timer
from src.score import validate_predictions

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
            logger.info(f"Training data processed: {train_processed.shape}")
            
            validation_processed = preprocessor.transform(validation_df)
            logger.info(f"Validation data processed: {validation_processed.shape}")
           
            
            logger.info("Feature Engineering")            
            feature_engineer = FeatureEngineer()
            
            train_features, feature_names = feature_engineer.fit_transform(train_processed)
            logger.info(f"Training features: {train_features.shape}")
            logger.info(f"Number of features: {len(feature_names)}")
            
            validation_features = feature_engineer.transform(validation_processed)
            logger.info(f"Validation features: {validation_features.shape}")
            
            # X and y for training
            feature_columns = [ 
                               column for column in train_features.columns 
                               if column not in (ID_COLUMN, TARGET_COLUMN) ]
            
            X_train = train_features[feature_columns].values 
            y_train = train_features[TARGET_COLUMN].values
            
            logger.info(f"Training matrix: {X_train.shape}")
            logger.info(f"Training targets: {y_train.shape}")
            
            
            
            logger.info("Training Model")
            
            trainer = ModelTrainer()
            
            if not args.predict_only:
                model = trainer.train(
                    X_train, y_train,
                    tune_hyperparameters=not args.skip_tune,
                    n_trials=args.n_trials
                )
                
                trainer.save_model()
                joblib.dump(feature_names, FEATURE_NAMES_PATH)
                logger.info(f"Feature names saved to {FEATURE_NAMES_PATH}")
            else:
                trainer.load_model()
                logger.info("Using existing model")
            
            
            logger.info("Evaluating Model")
            
            evaluator = ModelEvaluator()
            
            y_pred_train = trainer.model.predict(X_train)
            
            metrics = evaluator.evaluate(y_train, y_pred_train)
            
            # Residual ploting
            evaluator.plot_residuals(
                y_train, y_pred_train,
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
            
            validation_columns = [ column for column in validation_features.columns 
                                  if column != ID_COLUMN ]
            
            X_val = validation_features[validation_columns].values
            
            validation_predictions = predictor.predict_validation_data(
                X_val,
                validation_features[ID_COLUMN].values,
            )
            
            
            output_path = PREDICTIONS_DIR / 'validation_predictions.csv'
            predictor.save_predictions(validation_predictions, output_path)
            
            template_df['predicted_rate'] = validation_predictions['predicted_rate'].values
            template_output = PREDICTIONS_DIR / 'validation_predictions_template_filled.csv'
            predictor.save_predictions(template_df, template_output)
            
            
            
            if DECEMBER_CHART_FILE.exists():
                logger.info("Generating December Chart")
                december_data = pd.read_csv(DECEMBER_CHART_FILE)
                
                december_processed = preprocessor.transform(december_data)
                december_features = feature_engineer.transform(december_processed)
                
                X_dec = december_features[[col for col in december_features.columns if col not in [ID_COLUMN]]].values
                
                dec_predictions = predictor.predict(X_dec)
                
                dec_chart_df = december_data.copy()
                dec_chart_df['predicted_rate'] = dec_predictions
                
                dec_path = PREDICTIONS_DIR / 'december_predictions.csv'
                dec_chart_df.to_csv(dec_path, index=False)
                logger.info(f"December predictions saved to {dec_path}")
                
                
                
                plot_december_chart(
                    dec_chart_df,
                    december_data,
                    save_path=FIGURES_DIR / 'december_chart.png'
                )
            else:
                logger.warning(f"December chart file not found: {DECEMBER_CHART_FILE}")
            
            logger.info("Validating Final Outputs")
            
            is_valid = validate_predictions(
                validation_predictions,
                template_df
            )
            
            if is_valid:
                logger.info("All validations passed!")
            else:
                logger.warning("Some validations failed - please check the output")
            
            logger.info("Pipeline Completed")
            logger.info(f"Total execution time: {total_timer.seconds:.1f} seconds")
            logger.info(f"Model performance:")
            logger.info(f"RMSE: ${metrics['rmse']:.2f}")
            logger.info(f"MAE: ${metrics['mae']:.2f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            logger.info(f"MAPE: {metrics['mape']:.2f}%")
            logger.info(f"Output files:")
            logger.info(f"Predictions: {output_path}")
            logger.info(f"December chart: {FIGURES_DIR / 'december_chart.png'}")
            logger.info(f"Model: {MODEL_SAVE_PATH}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()