"""Suitability classification based on above-mean production."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import RANDOM_STATE, TEST_SIZE


class CropClassifier:
    """Fit a linear SVM to the repository's above-mean suitability definition."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models: dict[str, Pipeline] = {}
        self.test_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def train_svm(self, crop_col: str, features: Sequence[str]) -> dict[str, Any]:
        """Train and evaluate a scaled SVM without fitting transforms on test rows."""
        target = self.data[crop_col]
        y_binary = np.where(target >= target.mean(), 1, 0)
        class_counts = np.bincount(y_binary)
        if len(class_counts) < 2 or class_counts.min() < 2:
            logging.warning("At least two observations per class are required for %s.", crop_col)
            return {}

        X = pd.get_dummies(self.data[list(features)], drop_first=False, dtype=float)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_binary,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_binary,
        )
        train_class_counts = np.bincount(y_train)
        cv_folds = min(5, int(train_class_counts.min()))
        if cv_folds < 2:
            logging.warning("Training split is too small for cross-validation of %s.", crop_col)
            return {}

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
            ]
        )
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc",
        )
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)

        self.models[crop_col] = model
        self.test_data[crop_col] = (y_test, probabilities)
        return {
            "model": model,
            "auc": auc,
            "cv_auc_mean": cv_scores.mean(),
            "n_rows": len(X),
            "n_features": X.shape[1],
        }

    def plot_roc_curve(self, crop_col: str) -> None:
        """Show the retained test-split ROC curve for a fitted crop classifier."""
        if crop_col not in self.test_data:
            raise ValueError(f"No test data for {crop_col}; run training first.")

        y_test, probabilities = self.test_data[crop_col]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = roc_auc_score(y_test, probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC curve for {crop_col}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
