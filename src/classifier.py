"""
2024 Forestry Stats Smart Competition: Classifier Module
Trains classification models (SVM) to predict suitability.
Added cross-validation and visualization for elite portfolio standards.
"""

import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CropClassifier:
    """Trains classification models to predict suitability."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models = {}
        self.test_data = {}  # Store test data for plotting

    def train_svm(self, crop_col: str, features: List[str]) -> Dict:
        """Trains an SVM classifier for suitability prediction with CV."""
        # Convert to binary by mean
        mean_val = self.data[crop_col].mean()
        y_binary = np.where(self.data[crop_col] >= mean_val, 1, 0)

        if len(np.unique(y_binary)) < 2:
            logging.warning(
                f"Only one class present for {crop_col}. Skipping."
            )
            return {}

        X = self.data[features]
        # Handle categorical
        X = pd.get_dummies(X, drop_first=True)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42
        )

        model = SVC(kernel="linear", probability=True, random_state=42)

        # Elite addition: Cross-Validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc"
        )
        logging.info(
            f"Cross-Validation AUC scores for {crop_col}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        # Fit model
        model.fit(X_train, y_train)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)

        logging.info(f"Test Set Model for {crop_col}: AUC = {auc:.4f}")

        self.models[crop_col] = model
        self.test_data[crop_col] = (
            y_test,
            y_pred_prob,
        )  # Store for plotting

        return {"model": model, "auc": auc, "cv_auc_mean": cv_scores.mean()}

    def plot_roc_curve(self, crop_col: str):
        """Plots ROC curve for a specific crop."""
        if crop_col not in self.test_data:
            print(f"No test data for {crop_col}. Run training first.")
            return

        y_test, y_pred_prob = self.test_data[crop_col]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="blue", label=f"ROC Curve (AUC = {auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {crop_col}")
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    print("Classifier Module loaded.")
