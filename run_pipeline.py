"""
2024 Forestry Stats Smart Competition: Pipeline Runner
Ties all modules together to run the full analysis and modeling flow.
"""

import os
import sys

# Add src to path if running from root
sys.path.append(os.path.abspath("src"))

from src.classifier import CropClassifier
from src.data_loader import DataLoader
from src.predictor import CropPredictor
from src.stat_analyzer import StatAnalyzer


def main():
    print("==================================================")
    print("🌲 2024 Forestry Statistics Smart Competition 🌲")
    print("         Unified Analytics Pipeline Runner        ")
    print("==================================================")

    # Define paths - Using the path found in the notebooks
    data_path = "C:/Users/user/Desktop/진짜임 이게 찐.csv"

    if not os.path.exists(data_path):
        print(f"\n[⚠️ Warning] Data file not found at: {data_path}")
        print(
            "Please ensure the data file is placed in the correct location or update the path in this script."
        )
        print("Falling back to a simulated execution or stopping...")
        # For demonstration, we will stop. In a real scenario, you could load a sample.
        return

    # 1. Load Data
    print("\n[Step 1] Loading and Transforming Data...")
    loader = DataLoader(data_path)
    data = loader.load_and_transform()
    print("✓ Data loaded successfully with English column mapping.")

    # 2. Statistical Analysis
    print("\n[Step 2] Running Statistical Analysis...")
    analyzer = StatAnalyzer(data)
    # We'll analyze 'chestnut_kg' as a representative crop
    crop = "chestnut_kg"
    if crop in data.columns:
        chi2, anova, spearman = analyzer.perform_analysis(crop)
        print(f"✓ Statistical Analysis completed for {crop}.")
        print("\n--- Chi-Square Results ---")
        print(chi2)
        print("\n--- ANOVA Results ---")
        print(anova)
    else:
        print(f"Crop column '{crop}' not found in data.")

    # 3. Regression Modeling
    print("\n[Step 3] Training Regression Models...")
    predictor = CropPredictor(data)
    # Define features based on translated names
    features = [
        "avg_temp",
        "humidity",
        "precipitation",
        "soil_depth_type",
        "soil_texture_code",
    ]
    # Filter only existing features
    features = [f for f in features if f in data.columns]

    if crop in data.columns and features:
        print(f"Training Random Forest for {crop} using features: {features}")
        results = predictor.train_regression(crop, features)
        if results:
            print(f"✓ Model trained. R2 Score: {results['r2']:.4f}")
    else:
        print("Missing crop or features for regression.")

    # 4. Classification Modeling
    print("\n[Step 4] Training Classification Models...")
    classifier = CropClassifier(data)
    if crop in data.columns and features:
        print(
            f"Training SVM for {crop} suitability (binary classification)..."
        )
        cls_results = classifier.train_svm(crop, features)
        if cls_results:
            print(f"✓ Model trained. AUC Score: {cls_results['auc']:.4f}")
    else:
        print("Missing crop or features for classification.")

    print("\n==================================================")
    print("Pipeline execution completed.")
    print("==================================================")


if __name__ == "__main__":
    main()
