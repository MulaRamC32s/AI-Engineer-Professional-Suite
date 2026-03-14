import logging
import joblib
import pandas as pd
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Manages model training, hyperparameter tuning, and evaluation."""
    
    def __init__(self, model_save_path: str = "model.joblib"):
        self.model_save_path = model_save_path
        self.best_model = None
        
    def train_and_tune(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Trains a Random Forest with GridSearch tuning."""
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info("Initializing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        
        logger.info("Training models...")
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        
        self._evaluate(self.best_model, X_test, y_test)
        self._save_model()
        return self.best_model
        
    def _evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluates the trained model on test data."""
        logger.info("Evaluating model...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
    def _save_model(self):
        """Saves the trained model artifact."""
        if self.best_model:
            logger.info(f"Saving model to {self.model_save_path}")
            joblib.dump(self.best_model, self.model_save_path)
        else:
            logger.warning("No model to save. Train a model first.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = DataPipeline()
    X, y = pipeline.run_pipeline("dummy_path.csv")
    trainer = ModelTrainer()
    trainer.train_and_tune(X, y)
