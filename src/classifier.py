"""
2024 Forestry Stats Smart Competition: Classifier Module
Trains classification models (SVM) to predict suitability.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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

    def train_svm(self, crop_col: str, features: List[str]) -> Dict:
        """Trains an SVM classifier for suitability prediction."""
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
        model.fit(X_train, y_train)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)

        logging.info(f"Model for {crop_col}: AUC = {auc:.4f}")

        self.models[crop_col] = model

        return {"model": model, "auc": auc}


if __name__ == "__main__":
    print("Classifier Module loaded.")
