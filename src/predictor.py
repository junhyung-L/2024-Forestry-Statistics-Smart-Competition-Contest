"""
2024 Forestry Stats Smart Competition: Predictor Module
Trains regression models to predict crop production.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CropPredictor:
    """Trains regression models to predict crop production."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models = {}

    def train_regression(self, crop_col: str, features: List[str]) -> Dict:
        """Trains a RandomForestRegressor for a specific crop."""
        filtered_data = self.data[self.data[crop_col] > 0].copy()

        if len(filtered_data) < 10:
            logging.warning(f"Not enough data for {crop_col}. Skipping.")
            return {}

        X = filtered_data[features]
        y = filtered_data[crop_col]

        # Handle categorical variables in X if any
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        logging.info(f"Model for {crop_col}: R2 = {r2:.4f}, MSE = {mse:.4f}")

        self.models[crop_col] = model

        return {
            "model": model,
            "r2": r2,
            "mse": mse,
            "importances": dict(zip(X.columns, model.feature_importances_)),
        }


if __name__ == "__main__":
    print("Predictor Module loaded.")
