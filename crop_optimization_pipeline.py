"""
2024 Forestry Stats Smart Competition: Crop Optimization Pipeline
🏆 GRAND PRIZE (1st Place) Refined Version
Modular AI-driven Cultivation Suitability Analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import logging

# Configure Environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelBenchmarker:
    """Compare multiple models for crop-specific suitability prediction."""
    def __init__(self):
        self.models = {
            'SVM': SVC(probability=True),
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'LogisticRegression': LogisticRegression()
        }
        self.best_model = None

    def find_best_model(self, X_train, y_train, X_test, y_test):
        """Discovers the optimal model based on AUC score."""
        logging.info("Benchmarking models for crop suitability...")
        best_score = 0
        for name, clf in self.models.items():
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, probs)
            logging.info(f"Model: {name} | AUC: {score:.4f}")
            if score > best_score:
                best_score = score
                self.best_model = clf
        return self.best_model

class CultivationAdvisor:
    """Suggests the best-suited crop for a specific regional profile."""
    def __init__(self, trained_models: dict):
        self.trained_models = trained_models # {'Blackberry': model, 'Chestnut': model}

    def recommend(self, regional_profile: pd.Series):
        """Returns the crop with the highest predicted probability."""
        recommendations = {}
        for crop, model in self.trained_models.items():
            prob = model.predict_proba([regional_profile])[0][1]
            recommendations[crop] = prob
        
        best_crop = max(recommendations, key=recommendations.get)
        logging.info(f"Recommended Crop: {best_crop} (Suitability: {recommendations[best_crop]:.2f})")
        return best_crop

if __name__ == "__main__":
    # Portfolio Demo Logic
    logging.info("Forestry Suitability Engine initialized.")
    # In practice, we loop through all 7 primary crops to find the best-matched ROI.
    logging.info("Analyzing regional soil (pH, Moisture) and climate (Temp, Precip) variables...")
