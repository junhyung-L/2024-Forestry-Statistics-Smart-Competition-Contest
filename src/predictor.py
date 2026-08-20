"""Regression model for positive production observations."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TEST_SIZE


class CropPredictor:
    """Fit a random-forest regressor and report held-out and CV R-squared."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models: dict[str, Pipeline] = {}
        self.feature_importances: dict[str, dict[str, float]] = {}

    def train_regression(self, crop_col: str, features: Sequence[str]) -> dict[str, Any]:
        """Train on positive observations and return reproducible regression metrics.

        Missing values are imputed inside the sklearn pipeline, so each
        cross-validation fold and the held-out set are transformed independently.
        """
        filtered_data = self.data.loc[
            self.data[crop_col] > 0, list(features) + [crop_col]
        ].copy()
        if len(filtered_data) < 10:
            logging.warning("Not enough positive observations for %s; skipping regression.", crop_col)
            return {}

        X = pd.get_dummies(filtered_data[list(features)], drop_first=False, dtype=float)
        y = filtered_data[crop_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        if len(X_train) < 2:
            logging.warning("Training split is too small for %s; skipping regression.", crop_col)
            return {}

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)),
            ]
        )
        cv_folds = min(5, len(X_train))
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
            scoring="r2",
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        importances = dict(
            zip(X.columns, model.named_steps["model"].feature_importances_, strict=True)
        )
        self.models[crop_col] = model
        self.feature_importances[crop_col] = importances

        return {
            "model": model,
            "r2": r2_score(y_test, predictions),
            "mse": mean_squared_error(y_test, predictions),
            "cv_r2_mean": cv_scores.mean(),
            "importances": importances,
            "n_rows": len(filtered_data),
            "n_features": X.shape[1],
        }

    def plot_importances(self, crop_col: str) -> None:
        """Show the fitted random-forest impurity importances for one crop."""
        if crop_col not in self.feature_importances:
            raise ValueError(f"No importance data for {crop_col}; run training first.")

        ordered = sorted(
            self.feature_importances[crop_col].items(), key=lambda item: item[1], reverse=True
        )
        names, values = zip(*ordered, strict=True)
        plt.figure(figsize=(10, 6))
        plt.bar(names, values, color="skyblue")
        plt.title(f"Feature importances for {crop_col}")
        plt.ylabel("Impurity importance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
