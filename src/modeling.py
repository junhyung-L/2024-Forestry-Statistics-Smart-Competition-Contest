"""
2024 Forestry Stats Smart Competition: Modeling Module
Refactored from notebooks/final_modeling.ipynb
"""

import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ForestryModeler:
    """Trains regression models to predict crop production and extract feature importances."""

    def __init__(self, data_path: str, encoding: str = "cp949"):
        self.data_path = data_path
        self.encoding = encoding
        self.data = None
        self.models = {}
        self.feature_importances = {}

        # Define variables based on notebook analysis
        self.soil_vars = ["토성코드", "토양깊이유형", "토양형코드", "토양유효수분량"]
        self.climate_vars = [
            "연평균기온",
            "최고기온",
            "최저기온",
            "상대습도(%)",
            "강수량(mm)",
            "온도차이",
        ]
        self.crops = [
            "밤 (kg)",
            "복분자딸기 (kg)",
            "오갈피 (kg)",
            "더덕 (kg)",
            "생표고 (kg)",
        ]

    def load_data(self) -> pd.DataFrame:
        """Loads data and handles missing values."""
        logging.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path, encoding=self.encoding)

        # Create '온도차이' if not present
        if (
            "온도차이" not in self.data.columns
            and "최고기온" in self.data.columns
            and "최저기온" in self.data.columns
        ):
            self.data["온도차이"] = self.data["최고기온"] - self.data["최저기온"]

        # Handle missing values for climate variables
        for col in self.climate_vars:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].mean())

        return self.data

    def prepare_features(self, include_soil=True, include_climate=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepares feature matrix X and target matrix y."""
        if self.data is None:
            self.load_data()

        X_parts = []
        
        if include_climate:
            X_parts.append(self.data[self.climate_vars])
            
        if include_soil:
            # Convert categorical soil variables to dummy variables
            soil_dummies = pd.get_dummies(self.data[self.soil_vars], drop_first=True)
            X_parts.append(soil_dummies)
            
        X = pd.concat(X_parts, axis=1)
        y = self.data[self.crops]
        
        return X, y

    def train_model_for_crop(self, crop_col: str, include_soil=True, include_climate=True) -> Dict:
        """Trains a RandomForestRegressor for a specific crop."""
        if self.data is None:
            self.load_data()
            
        if crop_col not in self.crops:
            raise ValueError(f"Crop {crop_col} not in defined crops.")
            
        X, y = self.prepare_features(include_soil, include_climate)
        
        # Filter rows where production is not 0 (as done in notebooks)
        valid_idx = self.data[crop_col] > 0
        X_filtered = X[valid_idx]
        y_filtered = y[crop_col][valid_idx]
        
        if len(X_filtered) < 10:
            logging.warning(f"Not enough data for {crop_col}. Skipping.")
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        logging.info(f"Model for {crop_col}: R2 = {r2:.4f}, MSE = {mse:.4f}")
        
        self.models[crop_col] = model
        
        # Store feature importances
        importances = dict(zip(X_train.columns, model.feature_importances_))
        self.feature_importances[crop_col] = importances
        
        return {
            'model': model,
            'r2': r2,
            'mse': mse,
            'importances': importances
        }

    def train_all_models(self, include_soil=True, include_climate=True) -> Dict:
        """Trains models for all crops."""
        results = {}
        for crop in self.crops:
            logging.info(f"Training model for {crop}...")
            res = self.train_model_for_crop(crop, include_soil, include_climate)
            if res:
                results[crop] = res
        return results

    def plot_feature_importances(self, crop_col: str):
        """Plots feature importances for a specific crop."""
        if crop_col not in self.feature_importances:
            print(f"No importance data for {crop_col}. Run training first.")
            return
            
        importances = self.feature_importances[crop_col]
        sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.family'] = 'Malgun Gothic' # Support Korean
        plt.rcParams['axes.unicode_minus'] = False
        plt.bar(sorted_importances.keys(), sorted_importances.values(), color='skyblue')
        plt.title(f'Feature Importances for {crop_col}')
        plt.ylabel('Importance')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Forestry Modeling Module loaded.")
