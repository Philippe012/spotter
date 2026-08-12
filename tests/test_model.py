import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.train import ModelTrainer
from src.predict import Predictor

def test_model_initialization():
    trainer = ModelTrainer()
    assert trainer.cv_folds == 5
    assert trainer.random_state == 42
    print("Model initialization test passed")

def test_prediction_pipeline():
    X = np.random.randn(100, 10)
    y = np.random.randn(100) * 100 + 500
    
    trainer = ModelTrainer()
    model = trainer.train(X, y, tune_hyperparameters=False)
    
    predictor = Predictor(model)
    predictions = predictor.predict(X[:10])
    
    assert len(predictions) == 10
    assert predictions.shape == (10,)
    print("Prediction pipeline test passed")

if __name__ == "__main__":
    test_model_initialization()
    test_prediction_pipeline()