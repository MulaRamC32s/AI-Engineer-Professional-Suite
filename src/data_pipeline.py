import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DataPipeline:
    """Professional Data Pipeline for feature engineering and data cleaning."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def load_data(self, path: str) -> pd.DataFrame:
        """Loads data from a specified path."""
        try:
            logger.info(f"Loading data from {path}")
            # Mocking data load for template completeness
            data = pd.DataFrame({
                "feature1": np.random.randn(100),
                "feature2": np.random.rand(100) * 10,
                "category": np.random.choice(["A", "B", "C"], 100),
                "target": np.random.randint(0, 2, 100)
            })
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values and basic data cleaning."""
        logger.info("Starting data cleaning...")
        df_cleaned = df.copy()
        if "target" in df_cleaned.columns:
            df_cleaned = df_cleaned.dropna(subset=["target"])
        num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].median())
        cat_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        return df_cleaned
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs feature engineering like encoding and scaling."""
        logger.info("Starting feature engineering...")
        df_engineered = df.copy()
        if "category" in df_engineered.columns:
            df_engineered = pd.get_dummies(df_engineered, columns=["category"], drop_first=True)
        return df_engineered
        
    def run_pipeline(self, path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Executes the full pipeline and returns X, y."""
        df = self.load_data(path)
        df_cleaned = self.clean_data(df)
        df_final = self.engineer_features(df_cleaned)
        
        y = df_final["target"]
        X = df_final.drop(columns=["target"])
        return X, y

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = DataPipeline()
    X, y = pipeline.run_pipeline("dummy_path.csv")
    logger.info(f"Pipeline finished. X shape: {X.shape}, y shape: {y.shape}")
