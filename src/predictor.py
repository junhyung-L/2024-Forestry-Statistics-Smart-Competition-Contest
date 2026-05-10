"""
2024 Forestry Stats Smart Competition: Predictor Module
Trains regression models to predict crop production.
Added cross-validation and visualization for elite portfolio standards.
"""

import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CropPredictor:
    """Trains regression models to predict crop production."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models = {}
        self.feature_importances = {}

    def train_regression(self, crop_col: str, features: List[str]) -> Dict:
        """Trains a RandomForestRegressor for a specific crop with CV."""
        filtered_data = self.data[self.data[crop_col] > 0].copy()

        if len(filtered_data) < 10:
            logging.warning(f"Not enough data for {crop_col}. Skipping.")
            return {}

        X = filtered_data[features]
        y = filtered_data[crop_col]

        # Handle categorical variables in X if any
        X = pd.get_dummies(X, drop_first=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Elite addition: Cross-Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        logging.info(
            f"Cross-Validation R2 scores for {crop_col}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        # Fit model
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        logging.info(f"Test Set Model for {crop_col}: R2 = {r2:.4f}, MSE = {mse:.4f}")

        self.models[crop_col] = model
        self.feature_importances[crop_col] = dict(
            zip(X.columns, model.feature_importances_)
        )

        return {
            "model": model,
            "r2": r2,
            "mse": mse,
            "cv_r2_mean": cv_scores.mean(),
            "importances": self.feature_importances[crop_col],
        }

    def plot_importances(self, crop_col: str):
        """Plots feature importances for a specific crop."""
        if crop_col not in self.feature_importances:
            print(f"No importance data for {crop_col}. Run training first.")
            return

        importances = self.feature_importances[crop_col]
        sorted_importances = dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)
        )

        plt.figure(figsize=(10, 6))
        plt.bar(
            sorted_importances.keys(),
            sorted_importances.values(),
            color="skyblue",
        )
        plt.title(f"Feature Importances for {crop_col}")
        plt.ylabel("Importance")
        plt.xlabel("Features")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Predictor Module loaded.")
